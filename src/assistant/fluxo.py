# Fluxo automatizado com LangGraph. Atende o pedido do desafio: "ao receber
# informações sobre um paciente, o sistema aciona diferentes etapas: verificar
# exames pendentes, sugerir condutas e emitir alertas para a equipe médica".
#
# LangChain x LangGraph: a chain da Etapa 2 é uma linha reta (A -> B -> C).
# O LangGraph modela um GRAFO: tem estado compartilhado, nós (etapas) e
# arestas CONDICIONAIS (o caminho muda conforme o que aconteceu). É o que
# precisamos aqui: só alerta se houver exame pendente ou sugestão retida.
#
#   carregar_paciente --(não existe)--------------------------> registrar
#         |(existe)
#   verificar_exames -> sugerir_conduta --(há o que alertar)--> emitir_alerta -> registrar
#                                       --(nada a alertar)---> registrar

import json
import os
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.assistant import auditoria, prontuarios
from src.assistant.assistente import responder_seguro
from src.assistant.config import ALERTS_PATH, PERGUNTA_FLUXO


# O "estado" é um dicionário que passa de nó em nó. Cada nó lê o que precisa
# e devolve só as chaves que quer atualizar (o LangGraph junta tudo).
class EstadoFluxo(TypedDict, total=False):
    id_prontuario: str   # entrada do fluxo
    paciente: dict       # dados do paciente (None se não existir)
    exames_pendentes: list
    sugestao: dict       # resultado de responder_seguro (inclui guardrails)
    alertas: list        # alertas gerados para a equipe
    status: str          # "concluido" | "paciente_inexistente"


# ── Regras de alerta (num lugar só) ───────────────────────────────────
def _montar_alertas(estado):
    """
    Decide QUAIS alertas a equipe deve receber. Usada pela aresta condicional
    (para saber se há o que alertar) e pelo nó que emite, então os dois
    sempre concordam.
    """
    alertas = []

    pendentes = estado.get("exames_pendentes") or []
    if pendentes:
        itens = "; ".join(f"{e['tipo']} (pedido em {e['data_pedido']})" for e in pendentes)
        alertas.append({
            "tipo": "exames_pendentes",
            "mensagem": f"{len(pendentes)} exame(s) pendente(s): {itens}.",
        })

    sugestao = estado.get("sugestao") or {}
    if sugestao.get("status") not in (None, "respondida"):
        # A sugestão foi barrada pelos guardrails: um humano precisa olhar.
        alertas.append({
            "tipo": "sugestao_retida",
            "mensagem": (
                f"A sugestão do assistente não pôde ser liberada "
                f"({sugestao.get('codigo')}). Requer avaliação do médico responsável."
            ),
        })
    return alertas


# ── Nós do grafo (cada um é uma etapa do fluxo) ───────────────────────
def _carregar_paciente(estado):
    """Passo 1: busca o paciente no banco. Se não existir, o fluxo encerra."""
    paciente = prontuarios.buscar_paciente(estado["id_prontuario"])
    if paciente is None:
        return {"paciente": None, "status": "paciente_inexistente", "alertas": []}
    return {"paciente": paciente, "status": "concluido"}


def _verificar_exames(estado):
    """Passo 2: lista os exames que ainda não têm resultado."""
    return {"exames_pendentes": prontuarios.exames_pendentes(estado["id_prontuario"])}


def _fabricar_no_sugerir(chain):
    """
    Passo 3: pede ao assistente as próximas condutas. Passa por
    responder_seguro, então tem guardrails, explicação e auditoria.
    É uma "fábrica" porque o nó precisa da chain (o modelo carregado).
    """
    def _sugerir_conduta(estado):
        # Coloca o diagnóstico (e o estádio, quando há) na pergunta, para o
        # RAG buscar o protocolo certo para ESTE paciente.
        p = estado["paciente"]
        descricao = p["diagnostico"]
        if p.get("estadio") not in (None, "", "-"):
            descricao += f", estadio {p['estadio']}"
        pergunta = PERGUNTA_FLUXO.format(paciente=descricao)
        sugestao = responder_seguro(chain, pergunta, estado["id_prontuario"])
        return {"sugestao": sugestao}
    return _sugerir_conduta


def _emitir_alerta(estado):
    """
    Passo 4: envia os alertas à equipe. Aqui "enviar" = acrescentar em
    alertas.jsonl (uma fila simulada).
    """
    alertas = _montar_alertas(estado)
    os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(ALERTS_PATH, "a", encoding="utf-8") as f:
        for a in alertas:
            evento = {"timestamp": agora, "id_prontuario": estado["id_prontuario"],
                      "destino": "equipe médica", **a}
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return {"alertas": alertas}


def _registrar(estado):
    """Passo final: um evento-resumo do fluxo no log de auditoria."""
    sugestao = estado.get("sugestao") or {}
    alertas = estado.get("alertas") or []
    auditoria.registrar(
        bloqueado=bool(alertas),
        evento="fluxo_automatizado",
        id_prontuario=estado["id_prontuario"],
        status=estado.get("status"),
        exames_pendentes=len(estado.get("exames_pendentes") or []),
        alertas=[a["tipo"] for a in alertas],
        # Liga este resumo ao evento detalhado da sugestão no mesmo log.
        request_id_sugestao=sugestao.get("request_id"),
    )
    return {}


# ── Arestas condicionais (onde o caminho muda) ────────────────────────
def _paciente_existe(estado):
    return "continuar" if estado["status"] == "concluido" else "encerrar"


def _ha_o_que_alertar(estado):
    return "alertar" if _montar_alertas(estado) else "registrar"


def montar_fluxo(chain):
    """Monta e compila o grafo. O resultado tem .invoke({"id_prontuario": ...})."""
    grafo = StateGraph(EstadoFluxo)

    grafo.add_node("carregar_paciente", _carregar_paciente)
    grafo.add_node("verificar_exames", _verificar_exames)
    grafo.add_node("sugerir_conduta", _fabricar_no_sugerir(chain))
    grafo.add_node("emitir_alerta", _emitir_alerta)
    grafo.add_node("registrar", _registrar)

    grafo.add_edge(START, "carregar_paciente")
    grafo.add_conditional_edges(
        "carregar_paciente", _paciente_existe,
        {"continuar": "verificar_exames", "encerrar": "registrar"},
    )
    grafo.add_edge("verificar_exames", "sugerir_conduta")
    grafo.add_conditional_edges(
        "sugerir_conduta", _ha_o_que_alertar,
        {"alertar": "emitir_alerta", "registrar": "registrar"},
    )
    grafo.add_edge("emitir_alerta", "registrar")
    grafo.add_edge("registrar", END)

    return grafo.compile()

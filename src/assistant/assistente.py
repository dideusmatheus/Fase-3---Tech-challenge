# Ponto de entrada SEGURO do assistente. Junta as peças da Etapa 2 (chain) com
# as da Etapa 3 (guardrails + auditoria + explainability):
#
#   pergunta
#     -> guardrails de ENTRADA (regras, paciente existe?, está no escopo?)
#     -> chain da Etapa 2 (prontuário + RAG + LLM)
#     -> guardrails de SAÍDA (sem dose, sem prescrição, sem diagnóstico final)
#     -> explicação das fontes + aviso de validação humana
#     -> registro de auditoria
#
# Quem usa o assistente deve chamar responder_seguro(), e não chain.responder().

import re
import time
import uuid

from src.assistant import auditoria, chain as chain_mod, guardrails, prontuarios, rag
from src.assistant.config import ADAPTER_DIR, BASE_MODEL, TOP_K

# Citação de protocolo no texto da resposta, ex.: "PROT-01".
_CITACAO = re.compile(r"PROT-\d+")


def _explicar(resposta, fontes, id_prontuario, dados_paciente):
    """
    Explainability: mostra DE ONDE veio a resposta.
    - protocolos: quais protocolos entraram no contexto do modelo (RAG);
    - citou_fonte: o texto da resposta menciona algum protocolo? O modelo
      pequeno às vezes esquece de citar, então medimos em vez de supor;
    - prontuario: quais dados do paciente foram entregues ao modelo.
    """
    protocolos = sorted({f["protocolo"] for f in fontes})
    explicacao = {
        "protocolos": protocolos,
        "citou_fonte": bool(_CITACAO.search(resposta)),
        "prontuario": None,
    }
    if id_prontuario and dados_paciente:
        explicacao["prontuario"] = {
            "id": id_prontuario,
            "exames": len(dados_paciente["exames"]),
            "condutas": len(dados_paciente["condutas"]),
        }
    return explicacao


def _decidir_entrada(pergunta, id_prontuario):
    """Roda as 3 verificações de entrada e devolve a primeira que bloquear."""
    # 1) Regras de texto (prescrição, diagnóstico, injection, dados de terceiros).
    decisao = guardrails.verificar_entrada(pergunta)
    if not decisao["permitido"]:
        return decisao

    # 2) Se informou paciente, ele precisa existir no banco.
    if id_prontuario:
        existe = prontuarios.buscar_paciente(id_prontuario) is not None
        return guardrails.verificar_paciente(id_prontuario, existe)

    # 3) Sem paciente, a pergunta precisa ter relação com os protocolos.
    # (Com paciente não checamos: "quais exames estão pendentes?" fala do
    # prontuário e naturalmente não se parece com nenhum protocolo.)
    melhor_score = max(score for _, score in rag.buscar(pergunta, TOP_K))
    return guardrails.verificar_escopo(melhor_score)


def responder_seguro(chain, pergunta, id_prontuario=None):
    """
    Responde a pergunta aplicando todas as proteções.
    Devolve o mesmo dict de chain.responder(), mais:
      status:      "respondida" | "bloqueada_entrada" | "bloqueada_saida"
      codigo:      motivo do bloqueio (None quando respondida)
      request_id:  identificador do evento no log de auditoria
      explicacao:  fontes usadas (ver _explicar)
      aviso:       lembrete de validação humana (só quando respondida)
    """
    request_id = uuid.uuid4().hex[:8]
    inicio = time.perf_counter()
    resposta_bruta = None  # texto do modelo antes do bloqueio de saída (só vai ao log)

    entrada = _decidir_entrada(pergunta, id_prontuario)
    if not entrada["permitido"]:
        # Bloqueado ANTES do modelo: a LLM nem é chamada.
        resultado = {
            "pergunta": pergunta, "id_prontuario": id_prontuario,
            "resposta": entrada["mensagem"], "fontes": [], "dados_paciente": None,
            "status": "bloqueada_entrada", "codigo": entrada["codigo"],
            "explicacao": {"protocolos": [], "citou_fonte": False, "prontuario": None},
            "aviso": None,
        }
    else:
        resultado = chain_mod.responder(chain, pergunta, id_prontuario)
        resposta_bruta = resultado["resposta"]
        saida = guardrails.verificar_saida(resposta_bruta)
        resultado["explicacao"] = _explicar(
            resposta_bruta, resultado["fontes"], id_prontuario, resultado["dados_paciente"]
        )
        if saida["permitido"]:
            resultado.update(status="respondida", codigo=None, aviso=guardrails.AVISO_VALIDACAO)
        else:
            # O texto do modelo é descartado da resposta e fica só no log.
            resultado.update(
                status="bloqueada_saida", codigo=saida["codigo"],
                resposta=saida["mensagem"], aviso=None,
            )

    resultado["request_id"] = request_id

    # Auditoria: registra a decisão e as evidências, bloqueada ou não.
    auditoria.registrar(
        bloqueado=resultado["status"] != "respondida",
        request_id=request_id,
        pergunta=pergunta,
        id_prontuario=id_prontuario,
        status=resultado["status"],
        codigo=resultado["codigo"],
        fontes=resultado["fontes"],
        explicacao=resultado["explicacao"],
        resposta_entregue=resultado["resposta"],
        # Só preenchido quando a saída foi bloqueada: permite revisar o que o
        # modelo tentou dizer, sem entregar isso ao usuário.
        resposta_bruta=resposta_bruta if resultado["status"] == "bloqueada_saida" else None,
        modelo={"base": BASE_MODEL, "adaptador": ADAPTER_DIR},
        latencia_s=round(time.perf_counter() - inicio, 2),
    )
    return resultado

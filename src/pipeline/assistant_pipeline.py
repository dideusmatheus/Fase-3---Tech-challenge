import argparse
import json
import os

# Passos do pipeline, na ordem em que rodam. O passo "assistente" junta o
# LangChain (perguntas) e o LangGraph (fluxo por paciente).
ALL_STEPS = ["seed", "index", "assistente"]
# Partes do passo "assistente", para rodar uma de cada vez (ex.: no vídeo).
PARTES_ASSISTENTE = ["ask", "fluxo"]


def _tabela_validacao(resultados):
    """
    Resumo da validação: compara o que o assistente FEZ com o que DEVERIA
    fazer (campo "esperado" de cada pergunta). A pergunta avulsa não tem esse
    campo e fica de fora.
    """
    avaliadas = [r for r in resultados if r.get("decisao_correta") is not None]
    if not avaliadas:
        return []
    acertos = sum(r["decisao_correta"] for r in avaliadas)
    linhas = [
        f"**Validação das decisões:** {acertos}/{len(avaliadas)} corretas\n",
        "| Pergunta | Categoria | Esperado | Status | Motivo | Correta |",
        "|---|---|---|---|---|---|",
    ]
    for r in avaliadas:
        linhas.append(
            f"| {r['id']} | {r['categoria']} | {r['esperado']} | {r['status']} "
            f"| {r['codigo'] or '-'} | {'✅' if r['decisao_correta'] else '❌'} |"
        )
    linhas.append("")
    return linhas


def _linhas_fontes(r):
    """Explainability: de onde veio a resposta (protocolos e prontuário)."""
    exp = r["explicacao"]
    linhas = ["**Fontes (trechos usados no contexto):**"]
    linhas += [
        f"- {f['protocolo']} — {f['secao']} (similaridade {f['score']})" for f in r["fontes"]
    ]
    if not r["fontes"]:
        linhas.append("- nenhuma (a pergunta foi bloqueada antes de consultar o modelo)")
    if exp["prontuario"]:
        p = exp["prontuario"]
        linhas.append(f"- Prontuário {p['id']}: {p['exames']} exames e {p['condutas']} condutas no contexto")
    if r["status"] != "bloqueada_entrada":
        linhas.append(f"- Resposta cita o protocolo no texto: {'sim' if exp['citou_fonte'] else 'não'}")
    return linhas


def _bloco_pergunta(r):
    """Trecho do relatório referente a uma pergunta."""
    motivo = f" ({r['codigo']})" if r["codigo"] else ""
    linhas = [
        f"## {r.get('id', '-')} — {r['pergunta']}\n",
        f"- Paciente: {r['id_prontuario'] or 'nenhum'}",
        f"- Status: **{r['status']}**{motivo}",
        f"- ID de auditoria: `{r['request_id']}`",
        f"\n**Resposta:**\n\n{r['resposta']}\n",
    ]
    if r.get("aviso"):
        linhas.append(f"> {r['aviso']}\n")
    linhas += _linhas_fontes(r)
    linhas.append("")
    return linhas


def _salvar_relatorio(resultados, pasta):
    """Grava as respostas em JSON (para máquina) e Markdown (para leitura)."""
    os.makedirs(pasta, exist_ok=True)
    with open(f"{pasta}/respostas.json", "w", encoding="utf-8") as f:
        # ensure_ascii=False mantém acentos legíveis no arquivo.
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    linhas = ["# Respostas do assistente médico (Etapas 2 e 3)\n"]
    linhas += _tabela_validacao(resultados)
    for r in resultados:
        linhas += _bloco_pergunta(r)
    with open(f"{pasta}/respostas.md", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def _tabela_fluxo(resultados):
    """Resumo do fluxo: uma linha por paciente."""
    linhas = [
        "| Paciente | Status | Exames pendentes | Sugestão | Alertas |",
        "|---|---|---|---|---|",
    ]
    for r in resultados:
        sug = r["sugestao"]["status"] if r["sugestao"] else "-"
        alertas = ", ".join(a["tipo"] for a in r["alertas"]) or "nenhum"
        linhas.append(
            f"| {r['id_prontuario']} | {r['status']} | {len(r['exames_pendentes'])} "
            f"| {sug} | {alertas} |"
        )
    linhas.append("")
    return linhas


def _bloco_fluxo(r):
    """Trecho do relatório do fluxo referente a um paciente."""
    linhas = [f"## {r['id_prontuario']}\n"]
    if r["status"] == "paciente_inexistente":
        return linhas + ["Prontuário não encontrado: o fluxo foi encerrado.\n"]

    linhas.append("**Exames pendentes:**")
    linhas += [f"- {e['tipo']} (pedido em {e['data_pedido']})" for e in r["exames_pendentes"]]
    if not r["exames_pendentes"]:
        linhas.append("- nenhum")

    sug = r["sugestao"]
    linhas.append(f"\n**Sugestão do assistente** ({sug['status']}):\n\n{sug['resposta']}\n")

    linhas.append("**Alertas para a equipe médica:**")
    linhas += [f"- [{a['tipo']}] {a['mensagem']}" for a in r["alertas"]]
    if not r["alertas"]:
        linhas.append("- nenhum")
    linhas.append("")
    return linhas


def _salvar_relatorio_fluxo(resultados, pasta):
    """Grava o resultado do fluxo automatizado em JSON e Markdown."""
    os.makedirs(pasta, exist_ok=True)
    with open(f"{pasta}/fluxo.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    linhas = ["# Fluxo automatizado (LangGraph)\n"]
    linhas += _tabela_fluxo(resultados)
    for r in resultados:
        linhas += _bloco_fluxo(r)
    with open(f"{pasta}/fluxo.md", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def _rodar_perguntas(chain, pergunta, paciente, limit):
    """
    Parte LangChain do assistente: responde às perguntas fixas (ou a uma
    avulsa) com guardrails, explicação e auditoria. Devolve a lista de
    resultados, já com a avaliação "decisão correta" quando há gabarito.
    """
    from src.assistant.assistente import responder_seguro
    from src.assistant.config import QUESTIONS_PATH

    if pergunta:
        # Modo avulso: uma pergunta digitada no terminal.
        itens = [{"id": "AVULSA", "pergunta": pergunta, "id_prontuario": paciente}]
    else:
        with open(QUESTIONS_PATH, encoding="utf-8") as f:
            itens = json.load(f)
        if limit:
            itens = itens[:limit]

    resultados = []
    for item in itens:
        print(f"\n[{item['id']}] {item['pergunta']}")
        # responder_seguro aplica guardrails, explicação e auditoria.
        r = responder_seguro(chain, item["pergunta"], item.get("id_prontuario"))
        r["id"] = item["id"]
        r["categoria"] = item.get("categoria", "avulsa")
        # "esperado" diz o que o assistente DEVERIA fazer (responder ou
        # bloquear). A decisão é correta quando os dois batem.
        r["esperado"] = item.get("esperado")
        if r["esperado"] is None:
            r["decisao_correta"] = None
        else:
            r["decisao_correta"] = (r["status"] == "respondida") == (r["esperado"] == "responder")
        resultados.append(r)
        print(f"→ [{r['status']}] {r['resposta'][:300]}")
        if r["fontes"]:
            print("  fontes:", ", ".join(f"{x['protocolo']}" for x in r["fontes"]))
    return resultados


def _rodar_fluxo(chain, paciente):
    """
    Parte LangGraph do assistente: executa o grafo para um paciente (ou para
    todos, se None) e devolve uma lista de resultados simples, prontos para o
    relatório.
    """
    from src.assistant import prontuarios
    from src.assistant.fluxo import montar_fluxo

    ids = [paciente] if paciente else prontuarios.listar_ids()
    fluxo = montar_fluxo(chain)  # o grafo usa a MESMA chain das perguntas

    resultados = []
    for id_prontuario in ids:
        estado = fluxo.invoke({"id_prontuario": id_prontuario})
        sug = estado.get("sugestao")
        alertas = estado.get("alertas", [])
        resultados.append({
            "id_prontuario": id_prontuario,
            "status": estado["status"],
            "exames_pendentes": estado.get("exames_pendentes", []),
            "sugestao": {k: sug[k] for k in ("status", "codigo", "resposta", "request_id")} if sug else None,
            "alertas": alertas,
        })
        tipos = [a["tipo"] for a in alertas] or "nenhum"
        print(f"\n[{id_prontuario}] status={estado['status']} "
              f"pendentes={len(estado.get('exames_pendentes', []))} alertas={tipos}")
        if sug:
            print(f"  → [{sug['status']}] {sug['resposta'][:200]}")
    return resultados


def _passo_assistente(rodar_perguntas, rodar_fluxo, pergunta, paciente, limit):
    """
    Passo único do assistente: LangChain (perguntas) e LangGraph (fluxo)
    compartilham a mesma chain, então a LLM é carregada uma vez só.
    """
    from src.assistant.chain import montar_chain
    from src.assistant.config import ALERTS_PATH, AUDIT_PATH, REPORTS_DIR

    chain = montar_chain()

    if rodar_perguntas:
        resultados = _rodar_perguntas(chain, pergunta, paciente, limit)
        _salvar_relatorio(resultados, REPORTS_DIR)
        print(f"\n💾 Relatório salvo em {REPORTS_DIR}/respostas.md")
        print(f"🧾 Auditoria registrada em {AUDIT_PATH}")

    if rodar_fluxo:
        print("\n" + "-" * 55)
        print("🔀 Fluxo automatizado por paciente (LangGraph)")
        print("-" * 55)
        resultados_fluxo = _rodar_fluxo(chain, paciente)
        _salvar_relatorio_fluxo(resultados_fluxo, REPORTS_DIR)
        print(f"\n💾 Relatório salvo em {REPORTS_DIR}/fluxo.md")
        print(f"🚨 Alertas enviados à equipe em {ALERTS_PATH}")


def run_assistant_pipeline(steps=None, limit=None, pergunta=None, paciente=None):
    """
    Pipeline do assistente médico (Etapas 2 e 3 do desafio).

    Passos:
      seed       - cria o banco de prontuários fictícios (SQLite)
      index      - indexa os protocolos para a busca (RAG)
      assistente - LangChain + LangGraph juntos: responde às perguntas
                   (com guardrails e auditoria) e roda o fluxo automatizado

    Para rodar só uma parte do passo "assistente", use:
      ask   - só as perguntas (LangChain)
      fluxo - só o fluxo por paciente (LangGraph)

    Parâmetros:
      steps:    passos a rodar (padrão: seed, index e assistente).
      limit:    responde só às N primeiras perguntas fixas (teste rápido).
      pergunta: pergunta avulsa digitada no terminal (ignora o arquivo).
      paciente: id do prontuário para a pergunta avulsa e para o fluxo
                (sem ele, o fluxo percorre todos os pacientes).
    """
    steps = steps or ALL_STEPS

    # Imports pesados dentro de cada "if", como no pipeline da Etapa 1.
    if "seed" in steps:
        print("\n" + "=" * 55)
        print("🗄️  PASSO 1/3 — PRONTUÁRIOS FICTÍCIOS (SQLite)")
        print("=" * 55)
        from src.assistant.seed_prontuarios import seed_prontuarios
        seed_prontuarios()

    if "index" in steps:
        print("\n" + "=" * 55)
        print("🔎 PASSO 2/3 — INDEXAÇÃO DOS PROTOCOLOS (RAG)")
        print("=" * 55)
        from src.assistant.rag import indexar
        n = indexar()
        print(f"✅ {n} trechos de protocolo indexados")

    # "assistente" liga as duas partes. Com pergunta avulsa ele responde só a
    # ela (não faz sentido rodar o fluxo de todos os pacientes junto).
    rodar_perguntas = "assistente" in steps or "ask" in steps
    rodar_fluxo = "fluxo" in steps or ("assistente" in steps and not pergunta)
    if rodar_perguntas or rodar_fluxo:
        print("\n" + "=" * 55)
        print("🩺 PASSO 3/3 — ASSISTENTE MÉDICO (LangChain + LangGraph)")
        print("=" * 55)
        _passo_assistente(rodar_perguntas, rodar_fluxo, pergunta, paciente, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do assistente médico (Fase 3, Etapas 2 e 3)")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS + PARTES_ASSISTENTE, default=ALL_STEPS,
                        help="passos a rodar (padrão: seed index assistente); "
                             "ask e fluxo rodam só uma parte do passo assistente")
    parser.add_argument("--limit", type=int, default=None,
                        help="teste rápido: responde só às N primeiras perguntas")
    parser.add_argument("--pergunta", type=str, default=None,
                        help="faz uma pergunta avulsa em vez de usar o arquivo de perguntas")
    parser.add_argument("--paciente", type=str, default=None,
                        help="id do prontuário para a pergunta avulsa e para o fluxo "
                             "(ex.: PAC-0001); sem ele, o fluxo percorre todos os pacientes")
    args = parser.parse_args()
    run_assistant_pipeline(steps=args.steps, limit=args.limit,
                           pergunta=args.pergunta, paciente=args.paciente)

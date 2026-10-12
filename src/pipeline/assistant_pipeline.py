import argparse
import json
import os

# Etapas do pipeline, na ordem em que rodam.
ALL_STEPS = ["seed", "index", "ask"]


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


def run_assistant_pipeline(steps=None, limit=None, pergunta=None, paciente=None):
    """
    Pipeline da Etapa 2 (assistente médico com LangChain).

    Etapas:
      seed  - cria o banco de prontuários fictícios (SQLite)
      index - indexa os protocolos para a busca (RAG)
      ask   - responde às perguntas fixas (ou a uma pergunta avulsa)

    Parâmetros:
      steps:    etapas a rodar (padrão: todas).
      limit:    responde só às N primeiras perguntas fixas (teste rápido).
      pergunta: pergunta avulsa digitada no terminal (ignora o arquivo).
      paciente: id do prontuário para a pergunta avulsa (ex.: PAC-0001).
    """
    steps = steps or ALL_STEPS

    # Imports pesados dentro de cada "if", como no pipeline da Etapa 1.
    if "seed" in steps:
        print("\n" + "=" * 55)
        print("🗄️  ETAPA 1/3 — PRONTUÁRIOS FICTÍCIOS (SQLite)")
        print("=" * 55)
        from src.assistant.seed_prontuarios import seed_prontuarios
        seed_prontuarios()

    if "index" in steps:
        print("\n" + "=" * 55)
        print("🔎 ETAPA 2/3 — INDEXAÇÃO DOS PROTOCOLOS (RAG)")
        print("=" * 55)
        from src.assistant.rag import indexar
        n = indexar()
        print(f"✅ {n} trechos de protocolo indexados")

    if "ask" in steps:
        print("\n" + "=" * 55)
        print("🩺 ETAPA 3/3 — PERGUNTAS AO ASSISTENTE")
        print("=" * 55)
        from src.assistant.assistente import responder_seguro
        from src.assistant.chain import montar_chain
        from src.assistant.config import AUDIT_PATH, QUESTIONS_PATH, REPORTS_DIR

        if pergunta:
            # Modo avulso: uma pergunta digitada no terminal.
            itens = [{"id": "AVULSA", "pergunta": pergunta, "id_prontuario": paciente}]
        else:
            with open(QUESTIONS_PATH, encoding="utf-8") as f:
                itens = json.load(f)
            if limit:
                itens = itens[:limit]

        chain = montar_chain()  # carrega a LLM uma vez e reaproveita
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

        _salvar_relatorio(resultados, REPORTS_DIR)
        print(f"\n💾 Relatório salvo em {REPORTS_DIR}/respostas.md")
        print(f"🧾 Auditoria registrada em {AUDIT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline do assistente médico (Fase 3, Etapa 2)")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
                        help="etapas a rodar (padrão: todas)")
    parser.add_argument("--limit", type=int, default=None,
                        help="teste rápido: responde só às N primeiras perguntas")
    parser.add_argument("--pergunta", type=str, default=None,
                        help="faz uma pergunta avulsa em vez de usar o arquivo de perguntas")
    parser.add_argument("--paciente", type=str, default=None,
                        help="id do prontuário para a pergunta avulsa (ex.: PAC-0001)")
    args = parser.parse_args()
    run_assistant_pipeline(steps=args.steps, limit=args.limit,
                           pergunta=args.pergunta, paciente=args.paciente)

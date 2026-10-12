import argparse
import json
import os

# Etapas do pipeline, na ordem em que rodam.
ALL_STEPS = ["seed", "index", "ask"]


def _salvar_relatorio(resultados, pasta):
    """Grava as respostas em JSON (para máquina) e Markdown (para leitura)."""
    os.makedirs(pasta, exist_ok=True)
    with open(f"{pasta}/respostas.json", "w", encoding="utf-8") as f:
        # ensure_ascii=False mantém acentos legíveis no arquivo.
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    linhas = ["# Respostas do assistente médico (Etapa 2)\n"]
    for r in resultados:
        linhas.append(f"## {r.get('id', '-')} — {r['pergunta']}\n")
        linhas.append(f"- Paciente: {r['id_prontuario'] or 'nenhum'}")
        linhas.append(f"\n**Resposta:**\n\n{r['resposta']}\n")
        linhas.append("**Fontes (trechos usados no contexto):**")
        for f in r["fontes"]:
            linhas.append(f"- {f['protocolo']} — {f['secao']} (similaridade {f['score']})")
        linhas.append("")
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
        from src.assistant.chain import montar_chain, responder
        from src.assistant.config import QUESTIONS_PATH, REPORTS_DIR

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
            r = responder(chain, item["pergunta"], item.get("id_prontuario"))
            r["id"] = item["id"]
            resultados.append(r)
            print(f"→ {r['resposta'][:300]}")
            print("  fontes:", ", ".join(f"{x['protocolo']}" for x in r["fontes"]))

        _salvar_relatorio(resultados, REPORTS_DIR)
        print(f"\n💾 Relatório salvo em {REPORTS_DIR}/respostas.md")


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

# A chain do LangChain: junta prontuário + protocolos + LLM em um fluxo só.
# Usamos LCEL (LangChain Expression Language): os passos são ligados com "|",
# parecido com pipe do terminal — a saída de um vira a entrada do próximo.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.assistant import prontuarios, rag
from src.assistant.config import SYSTEM_PROMPT, TOP_K

# Molde da mensagem do médico. As chaves {paciente}, {trechos} e {pergunta}
# são preenchidas a cada chamada. O system prompt é o MESMO do treino.
TEMPLATE_HUMANO = (
    "Dados do paciente (prontuário):\n{paciente}\n\n"
    "Trechos dos protocolos do hospital:\n{trechos}\n\n"
    "Pergunta do médico: {pergunta}\n\n"
    "Responda com base nos trechos e nos dados acima, citando o protocolo "
    "usado (ex.: PROT-01). A decisão final é sempre do médico responsável."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", TEMPLATE_HUMANO),
])


def _buscar_contexto(entrada):
    """
    Passo 1 da chain: reúne o contexto da pergunta.
    entrada = {"pergunta": str, "id_prontuario": str | None}
    """
    texto_paciente, dados = None, None
    if entrada.get("id_prontuario"):
        texto_paciente, dados = prontuarios.resumo_paciente(entrada["id_prontuario"])

    # RAG: trechos de protocolo mais próximos da pergunta.
    achados = rag.buscar(entrada["pergunta"], TOP_K)
    return {
        "pergunta": entrada["pergunta"],
        "texto_paciente": texto_paciente or "Nenhum paciente informado.",
        "dados_paciente": dados,
        "trechos": [doc for doc, _ in achados],
        "scores": [float(score) for _, score in achados],
    }


def _para_prompt(ctx):
    """Passo 2: transforma o contexto nas 3 variáveis do template."""
    texto_trechos = "\n\n".join(
        f"[{d.metadata['protocolo']} — {d.metadata['secao']}]\n{d.page_content}"
        for d in ctx["trechos"]
    )
    return {
        "paciente": ctx["texto_paciente"],
        "trechos": texto_trechos,
        "pergunta": ctx["pergunta"],
    }


def montar_chain(llm=None):
    """
    Monta a chain completa:
      contexto -> (adiciona 'resposta' = prompt | LLM | texto)
    O resultado é um dict com pergunta, contexto usado e resposta.
    """
    if llm is None:
        from src.assistant.llm_local import carregar_llm
        llm = carregar_llm()

    gerar = RunnableLambda(_para_prompt) | PROMPT | llm | StrOutputParser()
    # RunnablePassthrough.assign mantém o dict de contexto e acrescenta a chave "resposta".
    return RunnableLambda(_buscar_contexto) | RunnablePassthrough.assign(resposta=gerar)


def responder(chain, pergunta, id_prontuario=None):
    """
    Executa a chain e devolve um dict simples, pronto para salvar em relatório.
    'fontes' é a explicação de onde veio a informação (usada na Etapa 3).
    """
    saida = chain.invoke({"pergunta": pergunta, "id_prontuario": id_prontuario})
    fontes = [
        {"protocolo": d.metadata["protocolo"], "secao": d.metadata["secao"], "score": round(s, 3)}
        for d, s in zip(saida["trechos"], saida["scores"])
    ]
    return {
        "pergunta": pergunta,
        "id_prontuario": id_prontuario,
        "resposta": saida["resposta"].strip(),
        "fontes": fontes,
        "dados_paciente": saida["dados_paciente"],
    }

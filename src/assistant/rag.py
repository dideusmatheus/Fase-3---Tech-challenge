# RAG = Retrieval-Augmented Generation (geração aumentada por busca).
# Ideia: antes de o modelo responder, buscamos nos protocolos do hospital os
# trechos mais parecidos com a pergunta e colocamos no prompt. Assim a
# resposta se apoia no protocolo (e não só na "memória" do modelo de 1,5B,
# que ainda erra fatos) e conseguimos informar a FONTE da informação.

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.assistant.config import EMBEDDING_MODEL, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, PROTOCOLS_DIR

# Guardamos o índice aqui depois de criado, para não refazer a cada pergunta.
_store = None


def _dividir_secao(texto, limite):
    """Se uma seção for grande demais, quebra em pedaços por linha."""
    if len(texto) <= limite:
        return [texto]
    pedacos, atual = [], ""
    for linha in texto.split("\n"):
        if atual and len(atual) + len(linha) > limite:
            pedacos.append(atual.strip())
            atual = ""
        atual += linha + "\n"
    if atual.strip():
        pedacos.append(atual.strip())
    return pedacos


def carregar_trechos():
    """
    Lê os PROT-XX.md e devolve uma lista de Document (um por seção).
    Cada Document tem o texto e metadados {protocolo, titulo, secao}: são
    esses metadados que viram a "fonte" mostrada na resposta.
    """
    docs = []
    for caminho in sorted(Path(PROTOCOLS_DIR).glob("PROT-*.md")):
        texto = caminho.read_text(encoding="utf-8")
        # A primeira linha é "# PROT-16 — Protocolo Interno: Manejo de ..."
        titulo = texto.split("\n", 1)[0].lstrip("# ").strip()
        protocolo = caminho.stem  # nome do arquivo sem .md => "PROT-16"

        # Divide pelas linhas que começam com "##", "###" ou "####" (cada seção).
        # Os protocolos não seguem o mesmo nível de título (uns usam "##", outros
        # "###"), então aceitamos os dois. O "# " simples é só o título do arquivo.
        # O [1:] descarta o que vem antes da primeira seção (cabeçalho).
        secoes = re.split(r"\n(?=#{2,4} )", texto)[1:]
        for secao in secoes:
            # Seção só com título e quase sem texto (ex.: "## Hospital Oncológico
            # Modelo") não ajuda a busca, então é ignorada.
            if len(secao.strip()) < MIN_CHUNK_CHARS:
                continue
            nome_secao = secao.split("\n", 1)[0].lstrip("# ").strip()
            for pedaco in _dividir_secao(secao, MAX_CHUNK_CHARS):
                docs.append(Document(
                    # Repetimos o título no texto: ajuda a busca, pois uma
                    # seção solta ("Fluxo de Manejo") não diz de que é.
                    page_content=f"{titulo}\n{pedaco}",
                    metadata={"protocolo": protocolo, "titulo": titulo, "secao": nome_secao},
                ))
    return docs


def indexar():
    """Cria (ou devolve) o índice vetorial em memória. Devolve o nº de trechos."""
    global _store
    if _store is None:
        docs = carregar_trechos()
        if not docs:
            raise FileNotFoundError(f"Nenhum protocolo encontrado em {PROTOCOLS_DIR}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        # InMemoryVectorStore guarda os vetores na RAM. Com ~21 protocolos
        # (algumas centenas de trechos) é rápido e dispensa banco vetorial.
        _store = InMemoryVectorStore.from_documents(docs, embeddings)
        _store.n_trechos = len(docs)
    return _store.n_trechos


def buscar(pergunta, k):
    """
    Devolve os k trechos mais parecidos com a pergunta, como lista de
    (Document, score). Score maior = mais parecido (similaridade de cosseno).
    """
    indexar()  # garante que o índice existe
    return _store.similarity_search_with_score(pergunta, k=k)

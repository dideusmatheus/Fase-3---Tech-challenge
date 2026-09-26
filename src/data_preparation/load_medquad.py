import pandas as pd

from src.data_preparation.config import RAW_CSV, SEED

# Tipos de exemplo. Guardamos isso em cada linha para (1) balancear o split
# treino/val/teste e (2) saber depois de onde veio cada exemplo.
TYPE_BREAST = "mama"
TYPE_CANCER = "cancer_geral"


def load_raw(path=RAW_CSV):
    """
    Lê o CSV original do MedQuAD e remove linhas inutilizáveis.

    O CSV tem 4 colunas: question, answer, source e focus_area (o assunto
    da pergunta, ex: "Breast Cancer"). Algumas linhas vêm sem resposta ou
    sem assunto — sem resposta não há o que ensinar ao modelo, então saem.
    """
    df = pd.read_csv(path)

    # dropna(subset=...) remove só as linhas em que essas colunas estão vazias.
    df = df.dropna(subset=["question", "answer", "focus_area"])
    return df.reset_index(drop=True)


def select_relevant(df):
    """
    Filtra só o que interessa ao nosso projeto (câncer de mama):

      1. MAMA: qualquer assunto que contenha "breast" (ex: "Breast Cancer",
         "Male Breast Cancer", "BRCA1 hereditary breast and ovarian cancer").
         Exceção: "Breastfeeding" (amamentação), que não tem relação com câncer.
      2. CÂNCER EM GERAL: todas as linhas da fonte "CancerGov" (Instituto
         Nacional do Câncer dos EUA). Só o câncer de mama tem poucos exemplos
         (~75); os demais cânceres dão volume e vocabulário oncológico.

    Adiciona a coluna "type" com TYPE_BREAST ou TYPE_CANCER.
    """
    # str.contains(..., case=False) procura o texto ignorando maiúsculas.
    is_breast = df["focus_area"].str.contains("breast", case=False)
    is_breastfeeding = df["focus_area"].str.contains("breastfeeding", case=False)
    is_breast = is_breast & ~is_breastfeeding

    is_cancergov = df["source"] == "CancerGov"

    selected = df[is_breast | is_cancergov].copy()
    # Se a linha é de mama, marca como mama (mesmo que venha do CancerGov).
    selected["type"] = TYPE_CANCER
    selected.loc[is_breast[selected.index], "type"] = TYPE_BREAST
    return selected.reset_index(drop=True)


def sample_for_smoke_test(df, limit):
    """
    Devolve só `limit` linhas (metade mama, metade câncer geral) para
    testar o pipeline rápido, sem gastar a API com o dataset inteiro.
    """
    half = max(1, limit // 2)
    parts = []
    for type_name in (TYPE_BREAST, TYPE_CANCER):
        subset = df[df["type"] == type_name]
        # min(...) evita erro se houver menos linhas do que o pedido.
        parts.append(subset.sample(min(half, len(subset)), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)

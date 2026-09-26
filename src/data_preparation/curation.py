import re

from src.data_preparation.config import LONG_ANSWER_CHARS, MIN_ANSWER_CHARS

# Regex (padrão de busca em texto) que encontra links, ex: https://... ou www...
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text):
    """Remove links e espaços/quebras de linha repetidos de um texto."""
    text = URL_PATTERN.sub("", text)
    # \s+ = qualquer sequência de espaços, tabs ou quebras de linha;
    # troca por UM espaço só.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_question(question):
    """
    Arruma o formato esquisito das perguntas do MedQuAD.

    Exemplos do CSV: "What is (are) Breast Cancer ?" e
    "Who is at risk for Breast Cancer? ?" — o " ?" sobrando é ruído.
    """
    question = clean_text(question)
    # Junta todos os "?" finais (com ou sem espaço entre eles) em um só.
    question = re.sub(r"(\s*\?)+$", "?", question)
    # Alguns começam em minúscula ("what research (or clinical trials)...").
    return question[:1].upper() + question[1:]


def curate(df):
    """
    Curadoria do dataset: limpa, remove duplicados e descarta o que não serve.

    Passos:
      1. Limpa texto de perguntas e respostas.
      2. Descarta respostas muito curtas (não ensinam nada).
      3. Remove RESPOSTAS repetidas: no MedQuAD a mesma resposta aparece
         em várias linhas (ex: fontes diferentes). Ficar com uma só evita
         o modelo decorar o mesmo texto e evita que a mesma resposta caia
         no treino E no teste (vazamento de dados).
      4. Marca respostas longas (coluna "needs_summary"), que serão
         resumidas na etapa de tradução.

    Retorna: (dataframe curado, dicionário com contagens de cada descarte).
    """
    stats = {"linhas_iniciais": len(df)}

    df = df.copy()
    df["question"] = df["question"].map(clean_question)
    df["answer"] = df["answer"].map(clean_text)

    df = df[df["answer"].str.len() >= MIN_ANSWER_CHARS]
    stats["apos_remover_respostas_curtas"] = len(df)

    # drop_duplicates mantém a primeira ocorrência de cada resposta.
    df = df.drop_duplicates(subset=["answer"])
    stats["apos_remover_respostas_repetidas"] = len(df)

    df["needs_summary"] = df["answer"].str.len() > LONG_ANSWER_CHARS
    stats["respostas_longas_a_resumir"] = int(df["needs_summary"].sum())

    return df.reset_index(drop=True), stats

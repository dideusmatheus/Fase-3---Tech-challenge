import re
from collections import Counter

# Cada item é (nome do tipo, padrão regex, texto que substitui o dado).
# A ORDEM importa: CPF vem antes de telefone porque os dois são sequências
# de números, e queremos que o CPF seja reconhecido primeiro.
#
# Os dados do nosso projeto são públicos (MedQuAD) ou fictícios, mas o PDF
# pede anonimização — e essa etapa protege o pipeline caso, no futuro,
# entrem textos reais do hospital.
PII_PATTERNS = [
    ("cpf", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # Telefone brasileiro: (11) 91234-5678, (11) 3456-7890, 11 91234-5678
    # ou 91234-5678. O padrão é rígido de propósito: um mais solto
    # confundiria intervalos de anos como "2010-2015" com telefone.
    (
        "telefone",
        re.compile(r"\(\d{2}\)\s?\d{4,5}-\d{4}|\b\d{2}\s9\d{4}-\d{4}|\b9\d{4}-\d{4}"),
        "[TELEFONE]",
    ),
    # Datas no formato 25/12/2024 ou 25-12-24.
    ("data", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATA]"),
    # Número de prontuário/matrícula: "prontuário 12345", "MRN: 998877".
    ("prontuario", re.compile(r"(?i)\b(?:prontu[aá]rio|matr[ií]cula|mrn)\s*[:#nº]*\s*\d+"), "[PRONTUARIO]"),
    # Nome depois de um título: "Dra. Ana Souza", "Sr. João da Silva".
    # Só pega palavras que começam com maiúscula (com "de/da/do" no meio).
    (
        "nome",
        re.compile(
            r"\b(?:Dr|Dra|Sr|Sra|Paciente)\.?:?\s+"
            r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+"
            r"(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*"
        ),
        "[NOME]",
    ),
]


def mask_text(text):
    """
    Troca dados pessoais do texto por marcadores (ex: "[CPF]").

    Retorna: (texto mascarado, Counter com quantas vezes cada tipo foi achado).
    """
    counts = Counter()
    for name, pattern, replacement in PII_PATTERNS:
        # subn() faz a troca E devolve quantas trocas foram feitas.
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] += n
    return text, counts


def anonymize_records(records):
    """
    Aplica mask_text em pergunta e resposta de cada registro (lista de
    dicionários com as chaves "question" e "answer").

    Retorna: (registros anonimizados, Counter total de itens mascarados).
    """
    total = Counter()
    cleaned = []
    for record in records:
        new_record = dict(record)  # copia, pra não alterar o original
        for field in ("question", "answer"):
            new_record[field], counts = mask_text(record[field])
            total.update(counts)
        cleaned.append(new_record)
    return cleaned, total

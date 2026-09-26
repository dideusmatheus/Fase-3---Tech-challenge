import os
import re
import json
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

# Reaproveitamos o cliente da Fase 2 (lê a ANTHROPIC_API_KEY do .env).
from src.llm_interpretation.client import ask_claude
from src.data_preparation.config import (
    CACHE_DIR,
    TRANSLATION_CACHE,
    SUMMARY_TARGET_CHARS,
)

# Quantas traduções rodam ao mesmo tempo. Mais threads = mais rápido, mas
# aumenta a chance de bater no limite de requisições da API.
MAX_WORKERS = 6
MAX_RETRIES = 4
# Respostas gigantes (algumas passam de 29 mil caracteres) são cortadas antes
# de ir pra API: o resumo não precisa de tudo, e assim o custo fica controlado.
MAX_INPUT_CHARS = 12000

SYSTEM_PROMPT_TRANSLATE = (
    "Você é um tradutor médico profissional (inglês → português do Brasil). "
    "Traduza com fidelidade, sem acrescentar nem inventar informações.\n"
    "Regras:\n"
    "- Mantenha siglas e nomes de genes/medicamentos como no original "
    "(ex: BRCA1, HER2, tamoxifeno).\n"
    "- Use terminologia médica usual no Brasil (ex: 'câncer de mama', "
    "'quimioterapia', 'ensaios clínicos').\n"
    "- Responda SOMENTE no formato:\n"
    "PERGUNTA: <pergunta traduzida>\n"
    "RESPOSTA: <resposta traduzida>"
)

SYSTEM_PROMPT_SUMMARIZE = SYSTEM_PROMPT_TRANSLATE.replace(
    "Traduza com fidelidade, sem acrescentar nem inventar informações.",
    "Traduza a pergunta e RESUMA a resposta em português, com fidelidade, "
    f"em no máximo {SUMMARY_TARGET_CHARS} caracteres, mantendo os pontos "
    "clínicos mais importantes. Não acrescente nem invente informações.",
)

# Lock = "cadeado": só uma thread por vez pode escrever no arquivo de cache,
# senão duas threads escreveriam ao mesmo tempo e misturariam as linhas.
_cache_lock = threading.Lock()


def _cache_key(question, answer):
    """Identificador único do texto (hash). Mesmo texto = mesma chave."""
    return hashlib.sha1(f"{question}||{answer}".encode("utf-8")).hexdigest()


def _load_cache():
    """Lê as traduções já salvas (uma por linha, em JSON) num dicionário."""
    cache = {}
    if os.path.exists(TRANSLATION_CACHE):
        with open(TRANSLATION_CACHE, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cache[item["key"]] = item
    return cache


def _parse_response(text):
    """Extrai pergunta e resposta do texto no formato PERGUNTA:/RESPOSTA:."""
    match = re.search(r"PERGUNTA:\s*(.+?)\s*RESPOSTA:\s*(.+)", text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _translate_one(row, cache):
    """
    Traduz UM par pergunta/resposta (ou usa o cache, se já foi traduzido).
    Devolve um dicionário com question/answer em português.
    """
    question, answer = row["question"], row["answer"]
    key = _cache_key(question, answer)
    if key in cache:
        return cache[key]

    system = SYSTEM_PROMPT_SUMMARIZE if row["needs_summary"] else SYSTEM_PROMPT_TRANSLATE
    user = f"PERGUNTA: {question}\nRESPOSTA: {answer[:MAX_INPUT_CHARS]}"

    parsed = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            parsed = _parse_response(ask_claude(system, user, max_tokens=2000))
            if parsed:
                break
        except RuntimeError:
            # Limite de requisições da API: espera um pouco (cada tentativa
            # espera mais que a anterior) e tenta de novo.
            time.sleep(5 * attempt)
    if not parsed:
        return None  # esse par será ignorado (e contado no relatório)

    result = {
        "key": key,
        "question": parsed[0],
        "answer": parsed[1],
        "summarized": bool(row["needs_summary"]),
    }
    with _cache_lock:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(TRANSLATION_CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    cache[key] = result
    return result


def translate_dataframe(df):
    """
    Traduz (e resume, se for longo) todas as linhas do dataframe curado.

    Retorna: (lista de registros em português, quantidade de falhas).
    Cada registro tem: question, answer, source, focus_area, type, summarized.
    """
    cache = _load_cache()
    rows = df.to_dict("records")

    # ThreadPoolExecutor roda várias traduções em paralelo — como cada
    # tradução passa quase todo o tempo esperando a API, threads ajudam muito.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda r: _translate_one(r, cache), rows))

    records, failures = [], 0
    for row, translated in zip(rows, results):
        if translated is None:
            failures += 1
            continue
        records.append(
            {
                "question": translated["question"],
                "answer": translated["answer"],
                "source": f"MedQuAD ({row['source']}) — {row['focus_area']}",
                "type": row["type"],
                "summarized": translated["summarized"],
            }
        )
    return records, failures

import os
import json
import random
import hashlib
from collections import Counter, defaultdict

from src.data_preparation.config import (
    PROCESSED_DIR,
    SEED,
    SPLIT_RATIOS,
    SYSTEM_PROMPT,
)
from src.data_preparation.load_medquad import load_raw, select_relevant, sample_for_smoke_test
from src.data_preparation.curation import curate
from src.data_preparation.translator import translate_dataframe
from src.data_preparation.synthetic_hospital_data import generate_all
from src.data_preparation.anonymizer import anonymize_records

# Respostas maiores que isso (em caracteres, já em português) ficam de fora:
# não cabem nos 1024 tokens do treino e seriam cortadas no meio.
MAX_FINAL_ANSWER_CHARS = 3500


def _group_id(answer):
    """ID do grupo = hash curto da resposta (respostas iguais = mesmo grupo)."""
    return "MQ-" + hashlib.sha1(answer.encode("utf-8")).hexdigest()[:10]


def split_by_group(records, ratios=SPLIT_RATIOS, seed=SEED):
    """
    Divide em treino / validação / teste SEM separar registros do mesmo grupo.

    Por que por grupo? Exemplos do mesmo protocolo (ou com a mesma resposta)
    são muito parecidos. Se um fosse pro treino e outro pro teste, o modelo
    "colaria" na prova e a nota do teste ficaria falsamente alta (vazamento).

    O sorteio é feito separadamente para cada "type" (mama, protocolo,
    faq...), garantindo que TODOS os tipos apareçam em treino, val e teste.
    """
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    by_type = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_type[record["type"]][record["group_id"]].append(record)

    for type_name, groups in by_type.items():
        group_ids = sorted(groups)  # sorted = ordem estável antes de embaralhar
        rng.shuffle(group_ids)

        total = sum(len(groups[g]) for g in group_ids)
        train_limit = ratios[0] * total
        val_limit = (ratios[0] + ratios[1]) * total

        counted = 0
        for group_id in group_ids:
            # Decide o destino pelo total acumulado ANTES de adicionar o grupo.
            if counted < train_limit:
                target = "train"
            elif counted < val_limit:
                target = "val"
            else:
                target = "test"
            splits[target].extend(groups[group_id])
            counted += len(groups[group_id])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def check_no_leakage(splits):
    """Garante que nenhum grupo e nenhuma resposta aparece em 2 splits."""
    seen_groups, seen_answers = {}, {}
    for split_name, records in splits.items():
        for record in records:
            for value, seen, label in (
                (record["group_id"], seen_groups, "grupo"),
                (record["answer"], seen_answers, "resposta"),
            ):
                if seen.setdefault(value, split_name) != split_name:
                    raise AssertionError(
                        f"Vazamento: {label} '{str(value)[:40]}' está em "
                        f"'{seen[value]}' e em '{split_name}'."
                    )


def to_chat_format(record):
    """
    Converte um registro para o formato que o treino espera:
      prompt     = mensagens que o modelo RECEBE (system + pergunta)
      completion = mensagem que o modelo deve APRENDER a gerar (resposta)
    Separar assim faz o treino calcular o erro só sobre a resposta, e não
    sobre o system prompt e a pergunta (que o modelo não precisa "aprender").
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ],
        "completion": [{"role": "assistant", "content": record["answer"]}],
        "source": record["source"],
        "type": record["type"],
        "group_id": record["group_id"],
    }


def build_dataset(limit=None):
    """
    Executa TODA a preparação de dados e grava train/val/test.jsonl.

    Fluxo:
      1. Carrega o MedQuAD e filtra mama + câncer geral (CancerGov).
      2. Curadoria: limpeza, remoção de duplicados, marca respostas longas.
      3. Tradução para português (Claude), resumindo as respostas longas.
      4. Gera dados sintéticos do hospital (protocolos, FAQ, modelos, recusas).
      5. Anonimização de todos os textos.
      6. Split por grupo + checagem de vazamento.
      7. Grava os arquivos + dataset_card.md.

    `limit`: usa só ~N linhas do MedQuAD e poucos sintéticos (teste rápido).
    """
    out_dir = f"{PROCESSED_DIR}/smoke" if limit else PROCESSED_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/6] Carregando e filtrando o MedQuAD...")
    df = select_relevant(load_raw())
    print(f"  {len(df)} linhas relevantes: {df['type'].value_counts().to_dict()}")
    if limit:
        df = sample_for_smoke_test(df, limit)

    print("\n[2/6] Curadoria...")
    df, curation_stats = curate(df)
    for name, value in curation_stats.items():
        print(f"  {name}: {value}")

    print(f"\n[3/6] Traduzindo {len(df)} pares para português (com cache)...")
    medquad_records, failures = translate_dataframe(df)
    print(f"  {len(medquad_records)} traduzidos, {failures} falharam.")
    for record in medquad_records:
        record["group_id"] = _group_id(record["answer"])

    print("\n[4/6] Gerando dados sintéticos do hospital...")
    synthetic_records = generate_all(limit)

    print("\n[5/6] Anonimizando...")
    all_records = medquad_records + synthetic_records
    all_records, pii_counts = anonymize_records(all_records)
    all_records = [r for r in all_records if len(r["answer"]) <= MAX_FINAL_ANSWER_CHARS]
    print(f"  Itens mascarados: {dict(pii_counts) or 'nenhum'}")

    print("\n[6/6] Dividindo em treino/validação/teste...")
    splits = split_by_group(all_records)
    check_no_leakage(splits)

    for split_name, records in splits.items():
        with open(f"{out_dir}/{split_name}.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(to_chat_format(record), ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(records)} exemplos")

    _write_dataset_card(out_dir, splits, curation_stats, failures, pii_counts)
    print(f"\nDataset salvo em {out_dir}/")
    return splits


def _write_dataset_card(out_dir, splits, curation_stats, failures, pii_counts):
    """Escreve o 'cartão do dataset': documenta origem, licença e contagens."""
    type_counts = {
        name: dict(Counter(r["type"] for r in records))
        for name, records in splits.items()
    }
    table = "\n".join(
        f"| {name} | {len(records)} | {type_counts[name]} |"
        for name, records in splits.items()
    )
    card = f"""# Dataset Card — Fine-tuning do assistente médico (Fase 3)

## Composição

| Split | Exemplos | Por tipo |
|---|---|---|
{table}

## Origem dos dados

1. **MedQuAD** (perguntas e respostas de sites do NIH) — filtrado para câncer de
   mama e para a fonte CancerGov; traduzido/resumido para português com
   Claude Haiku. Licença **CC BY 4.0**.
   Citação obrigatória: Ben Abacha A., Demner-Fushman D. *A Question-Entailment
   Approach to Question Answering.* BMC Bioinformatics, 2019.
2. **Dados sintéticos do hospital (fictícios)** — protocolos internos, FAQ de
   médicos, modelos de laudo/receita e exemplos de recusa segura, gerados
   com Claude. Não contêm dados de pessoas ou instituições reais.

## Processamento

- Curadoria: {curation_stats}
- Traduções que falharam e foram descartadas: {failures}
- Anonimização (itens mascarados): {dict(pii_counts) or 'nenhum encontrado'}
- Split 80/10/10 feito por grupo (respostas idênticas / mesmo protocolo ficam
  sempre no mesmo split) e verificado automaticamente contra vazamento.

## Limitações

- Traduções e textos sintéticos foram gerados por IA e não passaram por revisão
  médica completa; o assistente é apoio à decisão e não substitui o médico.
- O MedQuAD é conteúdo público de educação em saúde, não protocolo real de hospital.
"""
    with open(f"{out_dir}/dataset_card.md", "w", encoding="utf-8") as f:
        f.write(card)

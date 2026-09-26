import argparse

# Etapas do pipeline, na ordem em que rodam.
ALL_STEPS = ["data", "train", "evaluate"]


def run_finetuning_pipeline(steps=None, limit=None):
    """
    Pipeline completo da Etapa 1 (Fine-tuning).

    Etapas:
      data     - prepara o dataset (carrega, cura, traduz, anonimiza, divide)
      train    - treina o modelo com LoRA
      evaluate - compara modelo base x modelo fine-tunado

    Parâmetros:
      steps: lista com as etapas a rodar (padrão: todas, na ordem).
      limit: se informado, roda uma versão minúscula de tudo (poucos exemplos
             e poucos passos de treino) só para testar se o código funciona.
    """
    steps = steps or ALL_STEPS

    # Os imports ficam DENTRO de cada "if" de propósito: bibliotecas pesadas
    # (torch, transformers) demoram vários segundos para carregar, e assim
    # quem roda só a etapa "data" não paga esse tempo.
    if "data" in steps:
        print("\n" + "=" * 55)
        print("📦 ETAPA 1/3 — PREPARAÇÃO DOS DADOS")
        print("=" * 55)
        from src.data_preparation.build_dataset import build_dataset
        build_dataset(limit=limit)

    if "train" in steps:
        print("\n" + "=" * 55)
        print("🧠 ETAPA 2/3 — FINE-TUNING (LoRA)")
        print("=" * 55)
        from src.fine_tuning.train_lora import train
        train(smoke=bool(limit))

    if "evaluate" in steps:
        print("\n" + "=" * 55)
        print("📊 ETAPA 3/3 — AVALIAÇÃO (base x fine-tunado)")
        print("=" * 55)
        from src.fine_tuning.evaluate import evaluate
        evaluate(smoke=bool(limit))


if __name__ == "__main__":
    # argparse transforma o que digitamos no terminal (ex: --steps data)
    # em variáveis que o Python entende.
    parser = argparse.ArgumentParser(description="Pipeline de fine-tuning (Fase 3, Etapa 1)")
    parser.add_argument(
        "--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
        help="etapas a rodar (padrão: todas)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="modo teste rápido: usa só ~N exemplos e poucos passos de treino",
    )
    args = parser.parse_args()
    run_finetuning_pipeline(steps=args.steps, limit=args.limit)

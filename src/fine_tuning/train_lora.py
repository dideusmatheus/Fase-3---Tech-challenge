import os
import json

import torch
import matplotlib

# "Agg" desenha gráficos direto em arquivo, sem abrir janela (útil em scripts).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from trl import SFTConfig, SFTTrainer

from src.data_preparation.config import PROCESSED_DIR, SEED
from src.fine_tuning import config


def _load_data(data_dir):
    """
    Carrega train.jsonl e val.jsonl gerados pela preparação de dados.
    Cada linha tem "prompt" (system + pergunta) e "completion" (resposta).
    """
    files = {
        "train": f"{data_dir}/train.jsonl",
        "validation": f"{data_dir}/val.jsonl",
    }
    for path in files.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} não encontrado. Rode primeiro: "
                "python -m src.pipeline.finetuning_pipeline --steps data"
            )
    return load_dataset("json", data_files=files)


def _save_loss_curve(log_history, reports_dir):
    """Salva o gráfico de loss (erro) de treino e validação por passo."""
    train_points = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    eval_points = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]

    plt.figure(figsize=(8, 4.5))
    if train_points:
        plt.plot(*zip(*train_points), label="Treino")
    if eval_points:
        plt.plot(*zip(*eval_points), marker="o", label="Validação")
    plt.xlabel("Passo de treino")
    plt.ylabel("Loss (quanto menor, melhor)")
    plt.title("Curva de aprendizado do fine-tuning (LoRA)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/training_loss.png", dpi=150)
    plt.close()


def train(smoke=False):
    """
    Faz o fine-tuning do modelo base com LoRA e salva o "adaptador".

    O adaptador é só o pedaço treinado (poucos MB). Para usar o modelo
    depois, carregamos o modelo base + esse adaptador por cima.

    smoke=True: treino minúsculo (1 época, dados de teste rápido) só para
    verificar que tudo funciona, sem gastar tempo/GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA não encontrada. Confira se o torch foi instalado com "
            "suporte a CUDA (veja o requirements.txt)."
        )

    data_dir = config.SMOKE_DATA_DIR if smoke else PROCESSED_DIR
    adapter_dir = config.SMOKE_ADAPTER_DIR if smoke else config.ADAPTER_DIR
    reports_dir = config.SMOKE_REPORTS_DIR if smoke else config.REPORTS_DIR
    os.makedirs(reports_dir, exist_ok=True)

    # Fixa a semente aleatória: o treino fica reproduzível.
    set_seed(SEED)

    dataset = _load_data(data_dir)
    print(f"Treino: {len(dataset['train'])} exemplos | Validação: {len(dataset['validation'])}")

    # O tokenizer converte texto em números (tokens) que o modelo entende.
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    # Carrega o modelo base. bfloat16 usa metade da memória do formato
    # normal (float32) com pouca perda de qualidade.
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=f"{adapter_dir}/checkpoints",
        num_train_epochs=1 if smoke else config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",   # o aprendizado vai diminuindo suavemente até o fim
        warmup_steps=config.WARMUP_RATIO,  # float = fração do total de passos
        bf16=True,
        # Troca memória por tempo: recalcula partes do modelo em vez de
        # guardá-las. Necessário para caber nos 8 GB da GPU.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config.MAX_SEQ_LENGTH,
        # Avalia na validação e salva um checkpoint a cada época; no fim
        # mantém o MELHOR (menor loss de validação), não o último.
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5 if smoke else 10,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )
    trainer.model.print_trainable_parameters()

    trainer.train()

    # Salva só o adaptador LoRA (e o tokenizer, para usar junto).
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(f"{reports_dir}/training_log.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    _save_loss_curve(trainer.state.log_history, reports_dir)

    print(f"\nAdaptador salvo em {adapter_dir}/")
    print(f"Curvas e log salvos em {reports_dir}/")
    return trainer.state.log_history

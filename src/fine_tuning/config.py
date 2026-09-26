# Configurações do fine-tuning. Todos os hiperparâmetros ficam aqui pra
# facilitar experimentar (mudar um número) sem mexer na lógica do treino.

from src.data_preparation.config import PROCESSED_DIR, SEED

# ── Modelo ────────────────────────────────────────────────────────────
# Qwen2.5-1.5B-Instruct: modelo pequeno (1,5 bilhão de parâmetros) que já
# sabe seguir instruções e entende português. Em bf16 ocupa ~3 GB, então
# cabe com folga na GPU de 8 GB. "Instruct" = já foi ajustado para conversar.
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Onde salvar ───────────────────────────────────────────────────────
# O treino de teste rápido (smoke) usa pastas separadas pra não misturar
# com o treino de verdade.
ADAPTER_DIR = "models/fine_tuned/qwen2.5-1.5b-mama-lora"
REPORTS_DIR = "reports/fine_tuning"
SMOKE_ADAPTER_DIR = "models/fine_tuned/smoke"
SMOKE_REPORTS_DIR = "reports/fine_tuning/smoke"
SMOKE_DATA_DIR = f"{PROCESSED_DIR}/smoke"

# ── LoRA ──────────────────────────────────────────────────────────────
# LoRA NÃO treina o modelo inteiro (1,5 bi de números). Ele congela o
# modelo e treina só umas "matrizes pequenas" acopladas às camadas —
# poucos milhões de parâmetros. É bem mais leve e rápido, e não "esquece"
# o que o modelo já sabia.
LORA_R = 16              # tamanho das matrizes pequenas (maior = mais capacidade)
LORA_ALPHA = 32          # intensidade com que o ajuste é somado ao modelo (costuma ser 2x o r)
LORA_DROPOUT = 0.05      # "desliga" 5% das conexões ao treinar, para evitar decorar os dados
LORA_TARGET_MODULES = "all-linear"  # aplica o LoRA em todas as camadas lineares

# ── Treino ────────────────────────────────────────────────────────────
NUM_EPOCHS = 3            # quantas vezes o modelo vê o dataset inteiro
LEARNING_RATE = 2e-4      # tamanho do "passo" de aprendizado (LoRA aceita valores altos)
BATCH_SIZE = 2            # exemplos processados por vez na GPU (limitado pela memória)
GRAD_ACCUMULATION = 8     # acumula 8 lotes antes de atualizar => lote efetivo de 2 x 8 = 16
WARMUP_RATIO = 0.05       # nos primeiros 5% dos passos o aprendizado começa devagar
MAX_SEQ_LENGTH = 1024     # tamanho máximo (em tokens) de cada exemplo
EARLY_STOPPING_PATIENCE = 2  # para se a validação não melhorar por 2 épocas seguidas

# ── Avaliação ─────────────────────────────────────────────────────────
EVAL_MAX_NEW_TOKENS = 400     # tamanho máximo das respostas geradas na avaliação
EVAL_MAX_REFERENCE_CHARS = 1500  # a resposta de referência é cortada nesse tamanho para
                                 # comparar de forma justa com a resposta gerada (que é curta)
EVAL_BATCH_SIZE = 8

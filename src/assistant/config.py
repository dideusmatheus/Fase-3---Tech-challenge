# Configurações do assistente médico (Etapa 2).
# Igual aos outros config.py do projeto: caminhos e números ficam aqui, num
# lugar só, para não espalhar "strings mágicas" pelo código.

# Reaproveitamos o que a Etapa 1 já definiu: o modelo base, a pasta do
# adaptador LoRA treinado e o system prompt usado no treino.
from src.fine_tuning.config import ADAPTER_DIR, BASE_MODEL
from src.data_preparation.config import PROTOCOLS_DIR, SYSTEM_PROMPT, SEED

# ── Dados do assistente ───────────────────────────────────────────────
ASSISTANT_DATA_DIR = "data/assistant"
DB_PATH = f"{ASSISTANT_DATA_DIR}/prontuarios.db"      # banco SQLite com pacientes fictícios
QUESTIONS_PATH = f"{ASSISTANT_DATA_DIR}/perguntas.json"  # perguntas fixas do pipe
REPORTS_DIR = "reports/assistant"                      # onde o pipe grava as respostas

# ── RAG (busca nos protocolos) ────────────────────────────────────────
# Modelo de embeddings multilíngue: transforma texto em vetor de números.
# Textos com significado parecido ficam com vetores parecidos, e é assim
# que achamos o trecho do protocolo mais relevante para a pergunta.
# BGE-M3 (~2 GB, multilíngue). Testamos antes o MiniLM (bem menor), mas ele
# errava a busca em 2 de 6 perguntas de teste (ex.: não achava o PROT-21
# para "inteligência artificial"); o BGE-M3 acertou as 6.
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 3                 # quantos trechos de protocolo entram no prompt
MAX_CHUNK_CHARS = 1200   # trecho maior que isso é cortado (o treino foi com 1024 tokens)
MIN_CHUNK_CHARS = 120    # seção menor que isso (só título) é descartada

# ── Geração da resposta ───────────────────────────────────────────────
# Mesmos valores da avaliação da Etapa 1, para o comportamento ser o mesmo.
MAX_NEW_TOKENS = 400
REPETITION_PENALTY = 1.05  # penaliza repetir palavras
# do_sample=False => resposta determinística (mesma pergunta, mesma resposta).

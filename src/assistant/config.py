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
# Log de auditoria (Etapa 3). A pasta logs/ está no .gitignore de propósito:
# o log guarda as perguntas dos médicos, que não devem ir para o repositório.
LOGS_DIR = "logs/assistant"
AUDIT_PATH = f"{LOGS_DIR}/auditoria.jsonl"             # 1 evento JSON por linha

# ── Fluxo automatizado (LangGraph) ────────────────────────────────────
# "Fila" de alertas para a equipe médica. Simulamos o envio gravando em
# arquivo; num hospital real aqui entraria e-mail, sistema de mensagens etc.
ALERTS_PATH = f"{LOGS_DIR}/alertas.jsonl"
# Pergunta que o fluxo faz ao assistente para cada paciente. {paciente} é
# trocado pelo diagnóstico dela. Isso importa: o RAG busca pelo TEXTO da
# pergunta, e sem o diagnóstico ele trazia um protocolo qualquer (nos testes,
# o de recuperação pós-procedimento para todas as pacientes).
PERGUNTA_FLUXO = (
    "Quais as próximas condutas recomendadas pelos protocolos do hospital "
    "para uma paciente com {paciente}?"
)

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

# ── Segurança (Etapa 3) ───────────────────────────────────────────────
# Similaridade mínima do melhor trecho para a pergunta contar como "assunto do
# hospital". Medido nas nossas perguntas: as legítimas ficaram entre 0,54 e
# 0,71 e as fora de escopo (bolo, Copa do Mundo, poema...) entre 0,27 e 0,40.
# 0,45 fica no meio, com folga dos dois lados.
LIMIAR_ESCOPO = 0.45

# ── Geração da resposta ───────────────────────────────────────────────
# Mesmos valores da avaliação da Etapa 1, para o comportamento ser o mesmo.
MAX_NEW_TOKENS = 400
REPETITION_PENALTY = 1.05  # penaliza repetir palavras
# do_sample=False => resposta determinística (mesma pergunta, mesma resposta).

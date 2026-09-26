# Configurações da preparação de dados do fine-tuning.
# Centralizamos aqui caminhos e constantes pra não repetir "strings mágicas"
# em cada arquivo — se uma pasta mudar de nome, muda em um lugar só.
# Todos os caminhos são relativos à RAIZ do projeto (rode sempre com
# "python -m src...." a partir da raiz).

# Arquivo original do MedQuAD (baixado do Kaggle).
RAW_CSV = "data/fine_tuning/raw/medquad.csv"

# Onde guardamos as traduções já feitas. Traduzir custa chamadas de API;
# com esse cache, nunca pagamos duas vezes pelo mesmo texto.
CACHE_DIR = "data/fine_tuning/cache"
TRANSLATION_CACHE = f"{CACHE_DIR}/translations.jsonl"

# Dados fictícios do hospital (protocolos, FAQ, modelos de laudo/receita).
SYNTHETIC_DIR = "data/fine_tuning/synthetic"
PROTOCOLS_DIR = f"{SYNTHETIC_DIR}/protocols"

# Dataset final (train/val/test) que o treino consome.
PROCESSED_DIR = "data/fine_tuning/processed"

# Semente fixa: garante que o sorteio (split, amostras) seja sempre o mesmo
# e que os resultados possam ser reproduzidos.
SEED = 42

# Respostas maiores que isso (em caracteres) são RESUMIDAS na tradução,
# porque o treino tem limite de tamanho (1024 tokens por exemplo).
LONG_ANSWER_CHARS = 2000
# Tamanho aproximado do resumo gerado para as respostas longas.
SUMMARY_TARGET_CHARS = 1200
# Respostas muito curtas não ensinam nada ao modelo.
MIN_ANSWER_CHARS = 50

# Proporção do split treino / validação / teste.
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# System prompt usado em TODOS os exemplos de treino. É a "personalidade"
# e os limites do assistente; o mesmo texto será reutilizado no LangChain
# (Etapa 2) para o modelo se comportar igual no treino e no uso real.
SYSTEM_PROMPT = (
    "Você é um assistente médico virtual de apoio à equipe clínica de um "
    "hospital, com foco em oncologia e câncer de mama. Responda em português, "
    "de forma clara e objetiva.\n\n"
    "Regras:\n"
    "- Você NÃO prescreve medicamentos, doses ou condutas definitivas: "
    "qualquer sugestão deve ser validada pelo médico responsável.\n"
    "- Nunca afirme um diagnóstico com certeza absoluta.\n"
    "- Baseie-se nos protocolos do hospital e cite a fonte da informação "
    "quando ela for conhecida.\n"
    "- Se não souber ou a informação não estiver disponível, diga isso "
    "em vez de inventar."
)

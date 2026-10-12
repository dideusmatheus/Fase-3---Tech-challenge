# Cria o banco de prontuários FICTÍCIOS que o assistente vai consultar.
# O desafio pede "consultas em base de dados estruturada (prontuários e
# registros)". Como não temos dados reais do hospital (e nem poderíamos
# usar, por causa da LGPD), geramos pacientes inventados num SQLite.

import os
import sqlite3

from src.assistant.config import ASSISTANT_DATA_DIR, DB_PATH

# Estrutura das tabelas. Três tabelas ligadas pelo id_prontuario:
#   pacientes -> 1 linha por paciente
#   exames    -> vários exames por paciente (status pendente/concluido)
#   condutas  -> histórico do que a equipe decidiu
SCHEMA = """
CREATE TABLE pacientes (
    id_prontuario TEXT PRIMARY KEY,
    nome          TEXT NOT NULL,
    idade         INTEGER,
    diagnostico   TEXT,
    estadio       TEXT,
    tratamento    TEXT
);
CREATE TABLE exames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    id_prontuario TEXT NOT NULL REFERENCES pacientes(id_prontuario),
    tipo          TEXT NOT NULL,
    data_pedido   TEXT,
    status        TEXT NOT NULL CHECK (status IN ('pendente', 'concluido')),
    resultado     TEXT
);
CREATE TABLE condutas (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    id_prontuario TEXT NOT NULL REFERENCES pacientes(id_prontuario),
    data          TEXT,
    descricao     TEXT NOT NULL,
    responsavel   TEXT
);
"""

# Todos os nomes são inventados (sem relação com pessoas reais).
PACIENTES = [
    ("PAC-0001", "Maria Exemplo Silva", 54, "Carcinoma ductal invasivo de mama", "IIA", "Cirurgia + quimioterapia adjuvante"),
    ("PAC-0002", "Ana Teste Souza", 47, "Carcinoma ductal in situ", "0", "Aguardando cirurgia"),
    ("PAC-0003", "Joana Modelo Lima", 61, "Carcinoma lobular invasivo (RH+ / HER2-)", "IIB", "Hormonioterapia"),
    ("PAC-0004", "Carla Ficticia Rocha", 39, "Carcinoma de mama triplo negativo", "IIIA", "Quimioterapia neoadjuvante"),
    ("PAC-0005", "Lucia Sintetica Alves", 58, "Carcinoma de mama HER2+", "IIA", "Anti-HER2 + quimioterapia"),
    ("PAC-0006", "Paula Demo Costa", 66, "Nódulo mamário BI-RADS 4", "-", "Investigação diagnóstica"),
]

EXAMES = [
    # (id_prontuario, tipo, data_pedido, status, resultado)
    ("PAC-0001", "Mamografia bilateral", "2025-01-10", "concluido", "BI-RADS 5 em mama direita"),
    ("PAC-0001", "Ressonância magnética de mamas", "2025-03-02", "pendente", None),
    ("PAC-0001", "Ecocardiograma", "2025-03-05", "pendente", None),
    ("PAC-0002", "Biópsia por agulha grossa", "2025-02-12", "concluido", "CDIS grau intermediário"),
    ("PAC-0003", "Densitometria óssea", "2025-02-20", "pendente", None),
    ("PAC-0003", "Imunohistoquímica", "2025-01-15", "concluido", "RE+ RP+ HER2 negativo"),
    ("PAC-0004", "Hemograma", "2025-03-01", "pendente", None),
    ("PAC-0004", "Tomografia de estadiamento", "2025-02-18", "concluido", "Sem metástase à distância"),
    ("PAC-0005", "Ecocardiograma", "2025-02-25", "concluido", "FEVE 62%"),
    ("PAC-0006", "Biópsia guiada por ultrassom", "2025-03-04", "pendente", None),
]

CONDUTAS = [
    # (id_prontuario, data, descricao, responsavel)
    ("PAC-0001", "2025-01-20", "Discutido em reunião multidisciplinar; indicada cirurgia seguida de quimioterapia.", "Dr. Responsável A"),
    ("PAC-0002", "2025-02-20", "Encaminhada ao mastologista para planejamento cirúrgico.", "Dra. Responsável B"),
    ("PAC-0004", "2025-02-25", "Iniciado esquema neoadjuvante conforme decisão do oncologista.", "Dr. Responsável C"),
    ("PAC-0006", "2025-03-04", "Solicitada biópsia para esclarecer lesão BI-RADS 4.", "Dra. Responsável B"),
]


def seed_prontuarios():
    """Recria do zero o banco de prontuários fictícios."""
    os.makedirs(ASSISTANT_DATA_DIR, exist_ok=True)
    # Apaga o banco antigo: assim rodar o seed duas vezes dá sempre o mesmo resultado.
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)  # cria as 3 tabelas
    # executemany insere várias linhas de uma vez; os "?" são preenchidos
    # com os valores (evita montar SQL concatenando texto).
    con.executemany("INSERT INTO pacientes VALUES (?,?,?,?,?,?)", PACIENTES)
    con.executemany(
        "INSERT INTO exames (id_prontuario, tipo, data_pedido, status, resultado) VALUES (?,?,?,?,?)",
        EXAMES,
    )
    con.executemany(
        "INSERT INTO condutas (id_prontuario, data, descricao, responsavel) VALUES (?,?,?,?)",
        CONDUTAS,
    )
    con.commit()  # grava de verdade no arquivo
    con.close()
    print(f"✅ Banco criado em {DB_PATH}: {len(PACIENTES)} pacientes, "
          f"{len(EXAMES)} exames, {len(CONDUTAS)} condutas")


if __name__ == "__main__":
    seed_prontuarios()

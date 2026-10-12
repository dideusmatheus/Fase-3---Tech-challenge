# Consulta ao banco de prontuários (a parte "dados estruturados" do assistente).
# Regras de segurança deste módulo:
#   1. O banco é aberto SOMENTE LEITURA: o assistente nunca altera prontuário.
#   2. Todo SQL usa parâmetros "?": o id vem da pergunta do usuário e nunca
#      é concatenado no texto do SQL (evita SQL injection).

import sqlite3
from pathlib import Path

from src.assistant.config import DB_PATH


def _conectar():
    """Abre o banco em modo somente leitura (mode=ro)."""
    caminho = Path(DB_PATH).resolve()
    if not caminho.exists():
        raise FileNotFoundError(
            f"Banco não encontrado em {DB_PATH}. Rode: "
            "python -m src.pipeline.assistant_pipeline --steps seed"
        )
    # URI com mode=ro: qualquer INSERT/UPDATE aqui dá erro.
    con = sqlite3.connect(f"{caminho.as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row  # permite acessar colunas pelo nome
    return con


def buscar_paciente(id_prontuario):
    """Devolve os dados do paciente como dict, ou None se não existir."""
    con = _conectar()
    try:
        linha = con.execute(
            "SELECT * FROM pacientes WHERE id_prontuario = ?", (id_prontuario,)
        ).fetchone()
    finally:
        con.close()  # sempre fecha a conexão, mesmo se der erro
    return dict(linha) if linha else None


def listar_ids():
    """Ids de todos os prontuários (o fluxo automatizado percorre esta lista)."""
    con = _conectar()
    try:
        linhas = con.execute(
            "SELECT id_prontuario FROM pacientes ORDER BY id_prontuario"
        ).fetchall()
    finally:
        con.close()
    return [l["id_prontuario"] for l in linhas]


def listar_exames(id_prontuario, status=None):
    """Lista exames do paciente; se 'status' for informado, filtra por ele."""
    sql = "SELECT tipo, data_pedido, status, resultado FROM exames WHERE id_prontuario = ?"
    params = [id_prontuario]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY data_pedido"
    con = _conectar()
    try:
        linhas = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return [dict(l) for l in linhas]


def exames_pendentes(id_prontuario):
    """Atalho: só os exames que ainda não têm resultado."""
    return listar_exames(id_prontuario, status="pendente")


def ultimas_condutas(id_prontuario, limite=3):
    """As condutas mais recentes registradas pela equipe."""
    con = _conectar()
    try:
        linhas = con.execute(
            "SELECT data, descricao, responsavel FROM condutas "
            "WHERE id_prontuario = ? ORDER BY data DESC LIMIT ?",
            (id_prontuario, limite),
        ).fetchall()
    finally:
        con.close()
    return [dict(l) for l in linhas]


def resumo_paciente(id_prontuario):
    """
    Junta tudo num texto curto para entrar no prompt do modelo.
    Devolve (texto, dados) ou (None, None) se o paciente não existir.
    """
    paciente = buscar_paciente(id_prontuario)
    if paciente is None:
        return None, None

    exames = listar_exames(id_prontuario)
    condutas = ultimas_condutas(id_prontuario)

    linhas = [
        f"Prontuário: {paciente['id_prontuario']}",
        f"Idade: {paciente['idade']} anos",
        f"Diagnóstico: {paciente['diagnostico']} (estadio {paciente['estadio']})",
        f"Tratamento atual: {paciente['tratamento']}",
        "Exames:",
    ]
    for e in exames:
        resultado = e["resultado"] or "sem resultado"
        linhas.append(f"- {e['tipo']} ({e['status']}): {resultado}")
    if condutas:
        linhas.append("Últimas condutas:")
        for c in condutas:
            linhas.append(f"- {c['data']}: {c['descricao']}")

    dados = {"paciente": paciente, "exames": exames, "condutas": condutas}
    # Obs.: o NOME do paciente fica de fora do prompt de propósito. O modelo
    # não precisa dele para responder, e menos dado pessoal = menos risco.
    return "\n".join(linhas), dados

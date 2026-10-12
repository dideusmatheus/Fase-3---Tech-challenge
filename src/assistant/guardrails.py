# Guardrails = "grades de proteção" do assistente. Atendem o requisito
# "definir limites de atuação para evitar sugestões impróprias (ex.: nunca
# prescrever sem validação humana)".
#
# Por que em código e não só no system prompt? O modelo de 1,5B pode ignorar
# uma instrução do prompt (ou ser enganado por "ignore as suas regras"). Uma
# regra em código SEMPRE roda, não depende do que o modelo aprendeu.
#
# Duas camadas:
#   ENTRADA: checa a pergunta ANTES de chamar o modelo (barato e mais seguro:
#            o modelo nem chega a ver o pedido proibido).
#   SAÍDA:   checa a resposta DEPOIS de gerada (rede de proteção, caso o
#            modelo escorregue mesmo com uma pergunta aparentemente inocente).
#
# Este módulo só DECIDE (não lê banco, não chama modelo): fica fácil de testar.

import re
import unicodedata

from src.assistant.config import LIMIAR_ESCOPO

# Aviso que acompanha toda resposta liberada: a decisão final é sempre humana.
AVISO_VALIDACAO = (
    "⚠️ Apoio à decisão clínica: esta resposta não substitui a avaliação do "
    "médico responsável, que deve validar qualquer conduta."
)


def _normalizar(texto):
    """Minúsculas e sem acento, para os padrões não dependerem de 'é' x 'e'."""
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c)).lower()


def _liberado():
    return {"permitido": True, "codigo": None, "mensagem": None}


def _bloqueado(codigo, mensagem):
    return {"permitido": False, "codigo": codigo, "mensagem": mensagem}


# ── Camada de ENTRADA: padrões de pedidos proibidos ───────────────────
# Cada regra é (código, [expressões regulares], mensagem ao usuário).
# A ordem importa: a primeira que casar vence. "prompt_injection" vem primeiro
# porque costuma vir junto de outro pedido ("ignore as regras e prescreva...").
# Os padrões rodam sobre o texto já normalizado (sem acento, minúsculo).
REGRAS_ENTRADA = [
    ("prompt_injection", [
        r"\b(ignore|esqueca|desconsidere)\b.{0,40}\b(regras|instrucoes|restricoes)\b",
        r"\bsem restricoes\b",
        r"\bpule a validacao\b",
        r"\bsem validacao (medica|humana)\b",
        r"\bvoce agora e\b",
    ], "Não posso ignorar as regras de segurança do assistente. Posso ajudar "
       "com dúvidas sobre os protocolos do hospital e com os dados do prontuário."),

    ("prescricao_direta", [
        r"\b(prescreva|receite|prescreve)\b",
        r"\b(faca|escreva|emita|gere|redija)\b.{0,25}\b(receita|prescricao)\b",
        r"\bpode prescrever\b",
    ], "Não posso prescrever medicamentos ou doses. A prescrição é ato do médico "
       "responsável. Posso apoiar com o que os protocolos do hospital dizem sobre o caso."),

    ("diagnostico_definitivo", [
        r"\bconfirme\b.{0,40}\b(cancer|maligno|benigno|diagnostico|tumor|curada?)\b",
        r"\bdiga com certeza\b",
        r"\b(pode afirmar|afirme)\b",
        r"\bdiagnostico (final|definitivo)\b",
        r"\bsem (precisar de )?biopsia\b",
        r"\bso responda (sim|nao)\b",
    ], "Não posso afirmar um diagnóstico definitivo. O diagnóstico depende de "
       "exames e da avaliação do médico responsável."),

    ("alterar_tratamento", [
        r"\b(suspenda|interrompa|pare|aumente|reduza|diminua|troque|substitua)\b"
        r".{0,40}\b(quimio\w*|hormonio\w*|tratamento|dose|esquema|medicacao|terapia)\b",
    ], "Não posso alterar, suspender ou trocar um tratamento. Mudanças de "
       "conduta devem ser decididas e registradas pelo médico responsável."),

    ("dados_terceiros", [
        r"\b(cpf|rg|telefone|celular|endereco|e-?mail)\b",
        r"\b(liste|lista|todos)\b.{0,40}\b(nomes|pacientes)\b",
        r"\bprontuario completo\b",
    ], "Não posso fornecer dados pessoais identificáveis (CPF, contato, "
       "endereço) nem listas de pacientes."),
]


def verificar_entrada(pergunta):
    """
    Camada de entrada: devolve {"permitido", "codigo", "mensagem"}.
    Se casar com alguma regra proibida, a pergunta é bloqueada.
    """
    texto = _normalizar(pergunta)
    for codigo, padroes, mensagem in REGRAS_ENTRADA:
        if any(re.search(p, texto) for p in padroes):
            return _bloqueado(codigo, mensagem)
    return _liberado()


def verificar_paciente(id_prontuario, existe):
    """
    Se o médico informou um prontuário que não existe, NÃO deixamos o modelo
    responder: sem os dados reais ele poderia inventar exames e condutas.
    """
    if existe:
        return _liberado()
    return _bloqueado(
        "paciente_inexistente",
        f"Não encontrei o prontuário {id_prontuario}. Confira o identificador "
        "antes de continuar.",
    )


def verificar_escopo(melhor_score):
    """
    Pergunta sem paciente e sem nada parecido nos protocolos => fora do
    escopo do assistente (ex.: receita de bolo). O score é a similaridade do
    trecho mais próximo, vinda do RAG.
    """
    if melhor_score >= LIMIAR_ESCOPO:
        return _liberado()
    return _bloqueado(
        "fora_de_escopo",
        "Esta pergunta está fora do escopo do assistente. Só respondo sobre os "
        "protocolos do hospital e os prontuários dos pacientes.",
    )


# ── Camada de SAÍDA: o que a resposta gerada NÃO pode conter ──────────
# Dose (número + unidade) é a marca mais clara de prescrição. Somos
# conservadores de propósito: mesmo uma dose citada de um protocolo é
# bloqueada, porque o assistente nunca deve parecer estar prescrevendo.
# A resposta original continua no log de auditoria para revisão.
PADROES_SAIDA = [
    ("dose_na_resposta",
     r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|ug|g|ml|ui)\b",
     "A resposta gerada continha dose de medicamento e foi retida. Consulte o "
     "protocolo e o médico responsável para posologia."),
    ("prescricao_na_resposta",
     r"\b(prescrevo|receito|administre|inicie o uso)\b",
     "A resposta gerada tinha formato de prescrição e foi retida. A prescrição "
     "é ato do médico responsável."),
    ("diagnostico_definitivo_na_resposta",
     r"\b(com certeza absoluta|diagnostico definitivo|sem duvida alguma)\b",
     "A resposta gerada afirmava um diagnóstico definitivo e foi retida. O "
     "diagnóstico deve ser confirmado pelo médico responsável."),
]


def verificar_saida(resposta):
    """Camada de saída: confere o texto gerado pelo modelo."""
    texto = _normalizar(resposta)
    for codigo, padrao, mensagem in PADROES_SAIDA:
        if re.search(padrao, texto):
            return _bloqueado(codigo, mensagem)
    return _liberado()

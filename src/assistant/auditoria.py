# Auditoria = registro de TUDO que o assistente fez: quem perguntou o quê,
# o que foi decidido, quais fontes foram usadas. Atende o requisito
# "logging detalhado para rastreamento e auditoria".
#
# Formato JSON Lines (.jsonl): UM evento JSON por linha. É fácil de ler com
# o olho, de filtrar com grep e de carregar com pandas (read_json(lines=True)).
# O arquivo só é acrescentado (append), nunca reescrito: um log de auditoria
# não pode perder histórico.

import json
import logging
import os
from datetime import datetime, timezone

from src.assistant.config import AUDIT_PATH

# Logger criado uma vez e reaproveitado (como o modelo e o índice do RAG).
_logger = None


def _obter_logger():
    """Cria o logger de auditoria na primeira chamada."""
    global _logger
    if _logger is None:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        logger = logging.getLogger("assistente.auditoria")
        logger.setLevel(logging.INFO)
        # propagate=False: não repete a linha no console (que já tem os prints).
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(AUDIT_PATH, encoding="utf-8")
            # Só a mensagem: o próprio evento já traz o timestamp.
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        _logger = logger
    return _logger


def registrar(bloqueado, **campos):
    """
    Grava um evento de auditoria.
    bloqueado=True grava com nível WARNING (fácil de achar: só os bloqueios);
    bloqueado=False grava como INFO.
    """
    evento = {
        # UTC para o horário não depender do fuso da máquina.
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **campos,
    }
    nivel = logging.WARNING if bloqueado else logging.INFO
    # ensure_ascii=False mantém os acentos legíveis no arquivo.
    _obter_logger().log(nivel, json.dumps(evento, ensure_ascii=False))

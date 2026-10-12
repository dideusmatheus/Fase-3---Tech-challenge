# Carrega a LLM que NÓS treinamos na Etapa 1 (Qwen2.5-1.5B + adaptador LoRA)
# e a entrega num formato que o LangChain entende. É isso que atende o
# requisito "integrar a LLM customizada ao pipeline LangChain".

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.assistant.config import (
    ADAPTER_DIR, BASE_MODEL, MAX_NEW_TOKENS, REPETITION_PENALTY,
)

# Modelo carregado uma vez só (carregar leva alguns segundos e usa ~3 GB de GPU).
_llm = None


def carregar_llm():
    """Devolve o chat model do LangChain com o modelo fine-tunado."""
    global _llm
    if _llm is not None:
        return _llm

    if not torch.cuda.is_available():
        # Mesma exigência da Etapa 1: o modelo foi treinado/avaliado em GPU.
        raise RuntimeError("GPU CUDA não encontrada: o assistente usa o modelo em bf16 na GPU.")

    # O tokenizador do adaptador já traz o chat template do Qwen.
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    # 1) carrega o modelo base; 2) "encaixa" o adaptador LoRA por cima.
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    modelo = PeftModel.from_pretrained(base, ADAPTER_DIR).to("cuda").eval()

    # pipeline = objeto do transformers que junta tokenizador + modelo + geração.
    gerador = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,                       # sem sorteio: resposta reprodutível
        repetition_penalty=REPETITION_PENALTY,
        return_full_text=False,                # devolve só a resposta, sem repetir o prompt
    )
    # HuggingFacePipeline: adaptador do LangChain para esse pipeline.
    # ChatHuggingFace: aplica o chat template (system/human) automaticamente.
    _llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gerador))
    return _llm

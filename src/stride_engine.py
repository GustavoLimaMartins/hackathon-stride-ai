"""Motor de integração com o LLM para mapeamento STRIDE e prescrição de contramedidas.

Dois papéis de modelo: 'rewriter' (gpt-4o) interpreta/reescreve os dados brutos
do JSON estruturado antes da análise; 'analyst' (gpt-5) elabora o parecer STRIDE
final com as contramedidas.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

_MODELS = {
    "rewriter": "gpt-4o",
    "analyst": "gpt-5",
}

_llm_cache: dict[str, ChatOpenAI] = {}


def load_llm(role: str) -> ChatOpenAI:
    """Carrega (e cacheia) o cliente LLM correspondente ao papel informado.

    role: 'rewriter' (gpt-4o, interpreta os dados do JSON) ou 'analyst'
    (gpt-5, elabora o parecer STRIDE final).
    """
    if role not in _MODELS:
        raise ValueError(
            f"papel de LLM desconhecido: {role!r} (esperado um de {sorted(_MODELS)})"
        )

    if role in _llm_cache:
        return _llm_cache[role]

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY não configurada. Defina essa variável de ambiente "
            "(veja .env.example) antes de usar o stride_engine."
        )

    llm = ChatOpenAI(model=_MODELS[role])
    _llm_cache[role] = llm
    return llm

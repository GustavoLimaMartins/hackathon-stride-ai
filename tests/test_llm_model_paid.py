"""Testes da flag LLM_MODEL_PAID (provedor principal: OpenAI pago vs Ollama local).

Cobre, sem exigir OpenAI, Ollama ou rede:
  - Parsing de LLM_MODEL_PAID a partir de valores textuais de env (true/false/
    0/1/ausente), mesmo padrão de leitura usado por ANALYST_TIMEOUT_SECONDS.
  - load_primary_analyst() devolve OpenAIAnalyst quando a flag é verdadeira e
    OllamaAnalyst(is_fallback=False) quando é falsa.
  - OllamaAnalyst aceita is_fallback como atributo de instância (não mais fixo
    na classe), preservando o default True usado no fallback por timeout.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src import stride_engine
from src.stride_engine import (
    OllamaAnalyst,
    OllamaRewriter,
    OpenAIAnalyst,
    OpenAIRewriter,
    load_ollama_analyst,
    load_primary_analyst,
    load_primary_rewriter,
)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ],
)
def test_llm_model_paid_parses_env_values(raw_value, expected):
    """A leitura de LLM_MODEL_PAID interpreta variações textuais comuns."""
    parsed = raw_value.strip().lower() not in ("false", "0", "no")
    assert parsed is expected


def test_load_primary_analyst_returns_openai_when_paid():
    """LLM_MODEL_PAID=true (default): o principal é a OpenAI."""
    with patch.object(stride_engine, "_LLM_MODEL_PAID", True):
        analyst = load_primary_analyst()

    assert isinstance(analyst, OpenAIAnalyst)


def test_load_primary_analyst_returns_ollama_as_non_fallback_when_not_paid():
    """LLM_MODEL_PAID=false: o principal é o Ollama, e NÃO é um fallback."""
    with patch.object(stride_engine, "_LLM_MODEL_PAID", False):
        analyst = load_primary_analyst()

    assert isinstance(analyst, OllamaAnalyst)
    assert analyst.is_fallback is False


def test_load_ollama_analyst_defaults_to_fallback_true():
    """O uso existente no fallback por timeout (main.py) não muda: is_fallback=True."""
    analyst = load_ollama_analyst()
    assert analyst.is_fallback is True


def test_load_ollama_analyst_accepts_explicit_is_fallback_false():
    analyst = load_ollama_analyst(is_fallback=False)
    assert analyst.is_fallback is False


def test_flag_governs_both_roles_together():
    """LLM_MODEL_PAID controla analyst E rewriter de forma coerente."""
    with patch.object(stride_engine, "_LLM_MODEL_PAID", True):
        assert isinstance(load_primary_analyst(), OpenAIAnalyst)
        assert isinstance(load_primary_rewriter(), OpenAIRewriter)

    with patch.object(stride_engine, "_LLM_MODEL_PAID", False):
        analyst = load_primary_analyst()
        rewriter = load_primary_rewriter()
        assert isinstance(analyst, OllamaAnalyst) and analyst.is_fallback is False
        assert isinstance(rewriter, OllamaRewriter) and rewriter.is_fallback is False

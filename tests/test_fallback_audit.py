"""Testes do fallback local (OllamaAnalyst) e da trilha de auditoria (llm_audit).

Cobre, sem exigir OpenAI, Ollama ou rede:
  - OllamaAnalyst.analyze: com o ollama.Client mockado, devolve um StrideReport
    válido a partir do JSON do modelo, força o schema via format= e passa as
    options corretas.
  - llm_audit.record: grava uma linha JSON válida no arquivo e emite no logger.
  - llm_audit.audit_analysis: grava status 'success' no caminho feliz e
    'timeout'/'error' quando a chamada levanta, sempre re-levantando a exceção.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src import llm_audit
from src.stride_engine import OllamaAnalyst
from src.stride_models import StrideReport

# Um StrideReport mínimo, serializado como o Ollama devolveria em message.content.
_REPORT_JSON = json.dumps(
    {
        "risks": [
            {
                "target_type": "component",
                "target_id": "c0",
                "stride_category": "Spoofing",
                "elemento_afetado": "Usuário (c0)",
                "justificativa": "Ator externo sem autenticação forte na entrada.",
                "impacto": "Acesso indevido a componentes internos.",
                "severidade": "Alta",
                "contramedida": "Impor MFA e validação de identidade no gateway.",
            }
        ]
    }
)


def test_ollama_analyst_returns_structured_report(tmp_path):
    """OllamaAnalyst.analyze parseia o JSON do modelo num StrideReport válido."""
    fake_message = MagicMock()
    fake_message.content = _REPORT_JSON
    fake_response = MagicMock()
    fake_response.message = fake_message

    fake_client = MagicMock()
    fake_client.chat.return_value = fake_response

    # Redireciona a trilha de auditoria para um arquivo temporário e mocka o
    # Client do ollama (import tardio dentro de analyze).
    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"), patch(
        "ollama.Client", return_value=fake_client
    ):
        report = OllamaAnalyst().analyze("SYSTEM PROMPT", "USER MESSAGE")

    assert isinstance(report, StrideReport)
    assert len(report.risks) == 1
    assert report.risks[0].stride_category == "Spoofing"

    # O schema foi forçado e as options passadas.
    _, kwargs = fake_client.chat.call_args
    assert kwargs["model"] == OllamaAnalyst.model
    assert kwargs["format"] == StrideReport.model_json_schema()
    assert kwargs["options"]["temperature"] == 0
    assert kwargs["options"]["num_ctx"] > 0

    # A auditoria registrou o fallback com sucesso e nº de riscos.
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["provider"] == "ollama"
    assert entry["is_fallback"] is True
    assert entry["status"] == "success"
    assert entry["n_risks"] == 1
    assert "duration_s" in entry and "ts" in entry


def test_record_writes_jsonl_and_logs(tmp_path, caplog):
    """record() grava uma linha JSON no arquivo e emite no logger."""
    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"):
        with caplog.at_level(logging.INFO, logger="stride.llm_audit"):
            llm_audit.record({"provider": "openai", "model": "gpt-5", "status": "success"})

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["provider"] == "openai"
    assert "ts" in entry  # timestamp adicionado automaticamente
    assert any("llm_audit" in rec.message for rec in caplog.records)


def test_audit_analysis_success(tmp_path):
    """O contexto grava status 'success' e duração quando não há exceção."""
    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"):
        with llm_audit.audit_analysis("openai", "gpt-5", is_fallback=False) as audit:
            audit["n_risks"] = 3

    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip())
    assert entry["status"] == "success"
    assert entry["n_risks"] == 3
    assert entry["duration_s"] >= 0


def test_audit_analysis_timeout_is_classified_and_reraised(tmp_path):
    """Exceção com 'Timeout' no nome vira status 'timeout' e é re-levantada."""

    class APITimeoutError(Exception):
        pass

    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"):
        with pytest.raises(APITimeoutError):
            with llm_audit.audit_analysis("openai", "gpt-5", is_fallback=False):
                raise APITimeoutError("request timed out")

    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip())
    assert entry["status"] == "timeout"
    assert entry["error_type"] == "APITimeoutError"


def test_audit_analysis_error_is_classified_and_reraised(tmp_path):
    """Exceção genérica vira status 'error' e é re-levantada."""
    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"):
        with pytest.raises(ValueError):
            with llm_audit.audit_analysis("ollama", "gemma3:12b", is_fallback=True):
                raise ValueError("modelo ausente")

    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip())
    assert entry["status"] == "error"
    assert entry["error_type"] == "ValueError"
    assert entry["is_fallback"] is True

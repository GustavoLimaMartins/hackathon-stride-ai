"""Testes da etapa 'rewriter' (enriquecimento do grafo) e do papel dual-provider.

Cobre, sem exigir OpenAI, Ollama ou rede:
  - OllamaRewriter.rewrite / OpenAIRewriter.rewrite: com o cliente mockado,
    devolvem o grafo com 'narrative_summary' adicionado, preservando TODAS as
    chaves e ids originais (a rastreabilidade risco->bbox do analyst depende
    disso).
  - Auditoria: cada chamada grava provider/model/status/narrative_chars.
  - load_primary_rewriter() respeita LLM_MODEL_PAID (OpenAI vs Ollama), sem
    exigir OPENAI_API_KEY no modo local.
  - build_rewriter_prompt monta (system, user) a partir do grafo.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src import llm_audit, stride_engine
from src.prompts import build_rewriter_prompt
from src.stride_engine import (
    OllamaRewriter,
    OpenAIRewriter,
    load_ollama_rewriter,
    load_primary_rewriter,
)

# Grafo mínimo no formato de graph_builder.to_json(), com ids que NÃO podem
# sumir/mudar após o rewriter.
_GRAPH = {
    "trust_boundaries": [
        {
            "id": "b0",
            "label": "VPC",
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "components": [
                {
                    "id": "c0",
                    "name": "Amazon Lambda",
                    "class": "compute",
                    "bbox": [1.0, 1.0, 2.0, 2.0],
                    "confidence": 0.9,
                }
            ],
        }
    ],
    "unassigned_components": [],
    "data_flows": [{"source": "c0", "target": "c1", "confidence": 0.8, "arrow_bbox": [1, 1, 2, 2]}],
    "proximity_hints": [],
}


def _assert_structural_keys_intact(result: dict) -> None:
    """O rewriter só ADICIONA narrative_summary; nada estrutural muda."""
    for key in ("trust_boundaries", "unassigned_components", "data_flows", "proximity_hints"):
        assert result[key] == _GRAPH[key], f"chave estrutural {key} foi alterada"
    # Ids preservados (rastreabilidade risco->bbox).
    assert result["trust_boundaries"][0]["id"] == "b0"
    assert result["trust_boundaries"][0]["components"][0]["id"] == "c0"


def test_ollama_rewriter_adds_narrative_and_preserves_graph(tmp_path):
    """OllamaRewriter.rewrite devolve o grafo + narrative_summary, sem perder ids."""
    fake_message = MagicMock()
    fake_message.content = "Arquitetura de três camadas com um Lambda de computação."
    fake_response = MagicMock()
    fake_response.message = fake_message

    fake_client = MagicMock()
    fake_client.chat.return_value = fake_response

    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"), patch(
        "ollama.Client", return_value=fake_client
    ):
        result = OllamaRewriter().rewrite(_GRAPH)

    assert result["narrative_summary"].startswith("Arquitetura de três camadas")
    _assert_structural_keys_intact(result)

    # O rewriter NÃO força schema (é texto livre), diferente do analyst.
    _, kwargs = fake_client.chat.call_args
    assert kwargs["model"] == OllamaRewriter.model
    assert "format" not in kwargs
    assert kwargs["options"]["num_ctx"] > 0

    # Auditoria registrou o fallback (default) com sucesso e narrative_chars.
    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip())
    assert entry["provider"] == "ollama"
    assert entry["is_fallback"] is True
    assert entry["status"] == "success"
    assert entry["narrative_chars"] > 0


def test_openai_rewriter_adds_narrative_and_audits_tokens(tmp_path):
    """OpenAIRewriter.rewrite usa load_llm('rewriter') e audita provider=openai."""
    fake_ai_message = MagicMock()
    fake_ai_message.content = "  Resumo arquitetural.  "  # espaços devem ser tirados

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_ai_message

    with patch.object(llm_audit, "_AUDIT_PATH", tmp_path / "audit.jsonl"), patch.object(
        stride_engine, "load_llm", return_value=fake_llm
    ) as mock_load:
        result = OpenAIRewriter().rewrite(_GRAPH)

    mock_load.assert_called_once_with("rewriter")
    assert result["narrative_summary"] == "Resumo arquitetural."  # strip aplicado
    _assert_structural_keys_intact(result)

    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip())
    assert entry["provider"] == "openai"
    assert entry["model"] == OpenAIRewriter.model  # gpt-4o
    assert entry["is_fallback"] is False
    assert entry["status"] == "success"
    assert entry["narrative_chars"] == len("Resumo arquitetural.")


def test_load_primary_rewriter_openai_when_paid():
    with patch.object(stride_engine, "_LLM_MODEL_PAID", True):
        rewriter = load_primary_rewriter()
    assert isinstance(rewriter, OpenAIRewriter)


def test_load_primary_rewriter_ollama_non_fallback_when_not_paid():
    with patch.object(stride_engine, "_LLM_MODEL_PAID", False):
        rewriter = load_primary_rewriter()
    assert isinstance(rewriter, OllamaRewriter)
    assert rewriter.is_fallback is False


def test_load_ollama_rewriter_defaults_to_fallback_true():
    assert load_ollama_rewriter().is_fallback is True
    assert load_ollama_rewriter(is_fallback=False).is_fallback is False


def test_build_rewriter_prompt_returns_system_and_user():
    system_prompt, user_message = build_rewriter_prompt(_GRAPH)
    assert isinstance(system_prompt, str) and system_prompt
    assert "Amazon Lambda" in user_message  # o grafo foi serializado na mensagem
    assert "c0" in user_message


def test_rewriter_result_flows_into_analyst_user_message():
    """O narrative_summary do rewriter aparece na mensagem enviada ao analyst."""
    from src.prompts import build_stride_user_message

    enriched = {**_GRAPH, "narrative_summary": "PADRÃO-ARQUITETURAL-XYZ"}
    message = build_stride_user_message(enriched)
    assert "PADRÃO-ARQUITETURAL-XYZ" in message


def test_multiline_narrative_is_json_escaped_in_analyst_message():
    """Narrativa real (com quebras de linha e aspas) entra JSON-escapada.

    Reproduz o formato de saída real do modelo (parágrafos + aspas): a string
    crua NÃO aparece literalmente no JSON, mas sua forma escapada sim — que é o
    correto, pois o analyst recebe um JSON bem-formado.
    """
    import json

    from src.prompts import build_stride_user_message

    narrative = 'Arquitetura de três camadas.\nO "API Gateway" é o ponto de entrada.'
    enriched = {**_GRAPH, "narrative_summary": narrative}
    message = build_stride_user_message(enriched)

    assert narrative not in message  # a forma crua não aparece (tem \n e aspas)
    escaped = json.dumps(narrative, ensure_ascii=False)[1:-1]  # sem as aspas externas
    assert escaped in message  # a forma JSON-escapada, sim

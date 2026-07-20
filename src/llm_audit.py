"""Trilha de auditoria do processamento LLM da análise STRIDE.

Cada análise (parecer STRIDE gerado por um LLM) produz um registro de auditoria
que responde, para fins de rastreabilidade: QUAL provedor/modelo atendeu, se foi
o fallback local, quanto tempo levou, se teve sucesso/timeout/erro e o tamanho
do trabalho (tokens no caso OpenAI; tamanho do prompt e nº de riscos no fallback
Ollama, que não expõe tokens do mesmo jeito).

Os registros vão para dois destinos:
  1. o logger padrão "stride.llm_audit" (visível no console / integrável a
     qualquer handler de logging da aplicação);
  2. um arquivo JSONL append-only em logs/llm_audit.jsonl (uma linha JSON por
     análise) — a trilha persistente e máquina-legível.

Módulo puro (sem Streamlit) para ser testável isoladamente e reutilizável tanto
no caminho OpenAI quanto no fallback Ollama.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("stride.llm_audit")

# Arquivo da trilha de auditoria (uma linha JSON por análise). Fica sob logs/ na
# raiz do projeto; o diretório é criado sob demanda. logs/ é ignorado no git —
# é um artefato de runtime, não versionado.
_AUDIT_PATH = Path(__file__).resolve().parent.parent / "logs" / "llm_audit.jsonl"


def _utc_now_iso() -> str:
    """Timestamp UTC ISO-8601 — ordena a trilha e ancora cada registro no tempo."""
    return datetime.now(timezone.utc).isoformat()


def record(entry: dict[str, Any]) -> None:
    """Grava um registro de auditoria no logger e no arquivo JSONL.

    'entry' é o dicionário já montado do registro (provider, model, status,
    duration_s, tokens/tamanho...). Um campo 'ts' (timestamp UTC) é adicionado
    se ainda não estiver presente. A escrita em arquivo é resiliente: uma falha
    de I/O na trilha nunca deve derrubar a análise em si, então é apenas logada.
    """
    entry.setdefault("ts", _utc_now_iso())

    # Serialização estável (ensure_ascii=False p/ manter acentos legíveis).
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    logger.info("llm_audit %s", line)

    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        # Nunca propagar erro de I/O da trilha: a auditoria é observabilidade,
        # não deve quebrar a geração do parecer.
        logger.warning("falha ao gravar a trilha de auditoria em %s: %s", _AUDIT_PATH, exc)


@contextmanager
def audit_analysis(
    provider: str, model: str, is_fallback: bool
) -> Iterator[dict[str, Any]]:
    """Contexto que cronometra uma chamada de análise e grava a auditoria ao sair.

    Uso:
        with audit_analysis("openai", "gpt-5", is_fallback=False) as audit:
            report = ...          # chama o LLM
            audit["input_tokens"] = ...   # enriquece o registro com tokens/tamanho
            audit["output_tokens"] = ...
            audit["n_risks"] = len(report.risks)

    O dicionário cedido ('audit') começa com provider/model/is_fallback e é
    completado pelo chamador com os campos de tamanho. Ao sair do bloco, o
    contexto acrescenta duration_s e status e chama record():
      - sem exceção            -> status="success"
      - TimeoutError/APITimeout -> status="timeout" (reconhecido pelo nome da classe)
      - qualquer outra exceção  -> status="error", com error_type
    A exceção é sempre re-levantada (o contexto observa, não engole).
    """
    audit: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "is_fallback": is_fallback,
    }
    start = time.monotonic()
    try:
        yield audit
    except BaseException as exc:  # noqa: BLE001 — auditar qualquer falha e re-levantar
        audit["duration_s"] = round(time.monotonic() - start, 3)
        # "timeout" é reconhecido pelo nome da classe para não acoplar este
        # módulo puro ao SDK da OpenAI (APITimeoutError) nem ao concurrent.futures.
        name = type(exc).__name__
        audit["status"] = "timeout" if "Timeout" in name else "error"
        audit["error_type"] = name
        record(audit)
        raise
    else:
        audit["duration_s"] = round(time.monotonic() - start, 3)
        audit["status"] = "success"
        record(audit)

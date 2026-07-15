"""Testes da estimativa de progresso (src/progress.py).

O módulo é puro (sem Streamlit), então dá para exercitar todo o cálculo de
frações e a persistência dos tempos diretamente. O arquivo de calibração é
redirecionado para um tmp via monkeypatch para não tocar o do projeto.
"""

from __future__ import annotations

import json

import pytest

import src.progress as P
from src.progress import (
    DEFAULT_TIMINGS,
    STEP_KEYS,
    StepEstimator,
    format_elapsed,
    load_timings,
    save_timing,
)


@pytest.fixture
def tmp_timings(tmp_path, monkeypatch):
    """Redireciona o arquivo de tempos para um caminho temporário isolado."""
    path = tmp_path / ".stride_timings.json"
    monkeypatch.setattr(P, "_TIMINGS_PATH", path)
    return path


# --- StepEstimator ---------------------------------------------------------


def test_step_bounds_cover_full_bar_in_order():
    est = StepEstimator(DEFAULT_TIMINGS)
    # As fatias cobrem [0,1] sem buracos e em ordem.
    assert est.step_start(STEP_KEYS[0]) == 0.0
    assert est.step_end(STEP_KEYS[-1]) == pytest.approx(1.0)
    prev_end = 0.0
    for key in STEP_KEYS:
        assert est.step_start(key) == pytest.approx(prev_end)
        assert est.step_end(key) > est.step_start(key)
        prev_end = est.step_end(key)


def test_llm_gets_largest_slice():
    est = StepEstimator(DEFAULT_TIMINGS)
    widths = {k: est.step_end(k) - est.step_start(k) for k in STEP_KEYS}
    assert max(widths, key=widths.get) == "llm"


def test_fraction_never_exceeds_slice_before_completion():
    est = StepEstimator(DEFAULT_TIMINGS)
    # mesmo com elapsed >> eta, não passa do fim da fatia da etapa.
    for key in STEP_KEYS:
        assert est.fraction(key, 10_000) <= est.step_end(key)
        assert est.fraction(key, 0) == pytest.approx(est.step_start(key))


def test_fraction_is_monotonic_within_step():
    est = StepEstimator(DEFAULT_TIMINGS)
    seq = [est.fraction("llm", e) for e in (0, 10, 30, 60, 90)]
    assert seq == sorted(seq)
    assert all(0.0 <= f <= 1.0 for f in seq)


def test_step_percent_local_progress_and_cap():
    # ETA do llm nos defaults é 90s -> em 45s ~50%, e nunca 100% antes de concluir.
    est = StepEstimator(DEFAULT_TIMINGS)
    assert est.step_percent("llm", 0) == 0
    assert est.step_percent("llm", 45) == 50
    assert est.step_percent("llm", 90) == 99  # teto: só chega a 100 ao concluir
    assert est.step_percent("llm", 10_000) == 99


def test_slices_proportional_to_timings():
    # Duas etapas com tempos 1 e 3 -> fatias 25% e 75%.
    est = StepEstimator({"detect": 1, "ocr_zones": 3, "ocr_components": 0.0001,
                         "graph": 0.0001, "llm": 0.0001})
    w_detect = est.step_end("detect") - est.step_start("detect")
    w_zones = est.step_end("ocr_zones") - est.step_start("ocr_zones")
    assert w_zones == pytest.approx(3 * w_detect, rel=1e-3)


# --- persistência ----------------------------------------------------------


def test_load_timings_cold_start_returns_defaults(tmp_timings):
    assert load_timings() == DEFAULT_TIMINGS


def test_save_timing_applies_moving_average(tmp_timings):
    # EMA com alpha=0.5: 0.5*90 (default) + 0.5*30 = 60
    save_timing("llm", 30.0)
    assert load_timings()["llm"] == pytest.approx(60.0)
    assert tmp_timings.exists()


def test_save_timing_ignores_invalid(tmp_timings):
    save_timing("llm", 30.0)  # -> 60
    save_timing("desconhecida", 5.0)
    save_timing("llm", 0)
    save_timing("llm", -3)
    assert load_timings()["llm"] == pytest.approx(60.0)


def test_load_timings_corrupted_falls_back_to_defaults(tmp_timings):
    tmp_timings.write_text("{corrompido", encoding="utf-8")
    assert load_timings() == DEFAULT_TIMINGS


def test_load_timings_merges_partial_file_over_defaults(tmp_timings):
    tmp_timings.write_text(json.dumps({"llm": 120.0}), encoding="utf-8")
    t = load_timings()
    assert t["llm"] == 120.0
    assert t["detect"] == DEFAULT_TIMINGS["detect"]  # ausente -> default


# --- format_elapsed --------------------------------------------------------


def test_format_elapsed():
    assert format_elapsed(0) == "0:00"
    assert format_elapsed(5) == "0:05"
    assert format_elapsed(65) == "1:05"
    assert format_elapsed(125) == "2:05"
    assert format_elapsed(-10) == "0:00"  # negativo é saneado

"""Testes da consolidação por elemento e da padronização de imagem do relatório.

Cobre group_risks (um bloco por elemento, ordenação por severidade), o letterbox
(canvas fixo 600x400) e highlight_element (recorte padronizado de um grupo). Tudo
com imagem sintética e fixtures manuais — sem YOLO nem LLM.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from src.graph_builder import to_json
from src.stride_models import Risk, group_risks
from src.visual_report import (
    _CARD_HEIGHT,
    _CARD_WIDTH,
    highlight_element,
    standardize,
)

IMG = (1000, 1000)


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.fromarray(np.full((1000, 1000, 3), 255, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _graph() -> dict:
    tb = [{"class": "trust_boundary", "bbox": [0, 0, 900, 900], "confidence": 0.9, "label": "VPC"}]
    comps = [
        {"class": "api_gateway", "bbox": [100, 100, 140, 140], "confidence": 0.8, "name": "GW"},
        {"class": "database_storage", "bbox": [400, 400, 440, 440], "confidence": 0.8, "name": "DB"},
        {"class": "data_flow", "bbox": [150, 150, 390, 390], "confidence": 0.7, "name": ""},
    ]
    return to_json(tb, comps, image_size=IMG)


def _risk(tt, tid, cat, sev, fs=None, ft=None, elem="E") -> Risk:
    return Risk(
        target_type=tt, target_id=tid, flow_source_id=fs, flow_target_id=ft,
        stride_category=cat, elemento_afetado=elem, justificativa="j", impacto="i",
        severidade=sev, contramedida="c",
    )


# --- group_risks: consolidação --------------------------------------------


def test_same_component_multiple_categories_becomes_one_group():
    risks = [
        _risk("component", "c0", "Spoofing", "Média"),
        _risk("component", "c0", "Tampering", "Alta"),
        _risk("component", "c0", "Denial of Service", "Crítica"),
    ]
    groups = group_risks(risks)
    assert len(groups) == 1
    assert len(groups[0].risks) == 3
    # linhas ordenadas por severidade: Crítica -> Alta -> Média
    assert [r.severidade for r in groups[0].risks] == ["Crítica", "Alta", "Média"]
    # o representante é o de maior severidade
    assert groups[0].representative.severidade == "Crítica"


def test_flow_with_swapped_endpoints_is_one_group():
    risks = [
        _risk("flow", "c0", "Tampering", "Alta", fs="c0", ft="c1"),
        _risk("flow", "c1", "Spoofing", "Média", fs="c1", ft="c0"),  # invertido
    ]
    groups = group_risks(risks)
    assert len(groups) == 1
    assert len(groups[0].risks) == 2


def test_distinct_elements_stay_separate():
    risks = [
        _risk("component", "c0", "Spoofing", "Alta"),
        _risk("component", "c1", "Spoofing", "Alta"),
        _risk("boundary", "b0", "Information Disclosure", "Alta"),
    ]
    assert len(group_risks(risks)) == 3


def test_groups_sorted_by_worst_severity_first():
    risks = [
        _risk("component", "c0", "Spoofing", "Baixa"),
        _risk("component", "c1", "Spoofing", "Crítica"),
        _risk("component", "c2", "Spoofing", "Média"),
    ]
    groups = group_risks(risks)
    assert [g.representative.severidade for g in groups] == ["Crítica", "Média", "Baixa"]


def test_consolidation_reduces_block_count():
    # 6 riscos sobre 2 elementos -> 2 blocos (redução).
    risks = [_risk("component", "c0", c, "Alta") for c in ("Spoofing", "Tampering", "Repudiation")]
    risks += [_risk("component", "c1", c, "Média") for c in ("Spoofing", "DoS", "Elevation")]
    groups = group_risks(risks)
    assert len(groups) == 2
    assert sum(len(g.risks) for g in groups) == 6


# --- standardize / letterbox: canvas fixo ----------------------------------


def test_standardize_always_returns_card_size():
    for h, w in [(100, 900), (900, 100), (400, 400), (37, 512)]:
        out = standardize(np.full((h, w, 3), 120, dtype=np.uint8))
        assert out.shape == (_CARD_HEIGHT, _CARD_WIDTH, 3)


def test_standardize_pads_with_white_border():
    # recorte largo -> sobra padding branco em cima/embaixo.
    out = standardize(np.full((100, 900, 3), 10, dtype=np.uint8))
    assert (out[0] == 255).all() and (out[-1] == 255).all()


# --- highlight_element: recorte padronizado de um grupo --------------------


def test_highlight_element_returns_card_sized_image():
    graph = _graph()
    group = group_risks([_risk("component", "c0", "Spoofing", "Alta", elem="GW")])[0]
    out = highlight_element(_png_bytes(), group, graph)
    assert out is not None and out.shape == (_CARD_HEIGHT, _CARD_WIDTH, 3)


def test_highlight_element_invalid_target_returns_none():
    graph = _graph()
    group = group_risks([_risk("component", "c9999", "Spoofing", "Alta")])[0]
    assert highlight_element(_png_bytes(), group, graph) is None

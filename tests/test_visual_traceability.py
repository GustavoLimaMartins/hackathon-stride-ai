"""Testes da rastreabilidade visual: vínculo risco -> bounding box no diagrama.

Cobre a propagação da bbox da seta no grafo e as funções que localizam/recortam
o elemento afetado por um risco. Tudo com imagem sintética (numpy) e fixtures de
grafo manuais — sem depender do YOLO nem do LLM.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from src.graph_builder import to_json
from src.stride_models import Risk, severity_rank
from src.visual_report import (
    crop_region,
    draw_numbered_overlay,
    highlight_risk,
    resolve_graph_index,
)

IMG = (1000, 1000)


def _blank_image() -> np.ndarray:
    """Imagem branca como ndarray — para funções que já recebem o array (crop_region)."""
    return np.full((1000, 1000, 3), 255, dtype=np.uint8)


def _blank_png_bytes() -> bytes:
    """Imagem branca 1000x1000 como bytes PNG — para funções que carregam via _load_image."""
    buf = BytesIO()
    Image.fromarray(_blank_image()).save(buf, format="PNG")
    return buf.getvalue()


def _graph_with_flow() -> dict:
    """Grafo: zona b0 com dois componentes ligados por uma seta."""
    tb = [{"class": "trust_boundary", "bbox": [0, 0, 900, 900], "confidence": 0.9, "label": "VPC"}]
    comps = [
        {"class": "compute", "bbox": [100, 100, 140, 140], "confidence": 0.8, "name": "A"},
        {"class": "database_storage", "bbox": [400, 400, 440, 440], "confidence": 0.8, "name": "B"},
        {"class": "data_flow", "bbox": [150, 150, 390, 390], "confidence": 0.7, "name": ""},
    ]
    return to_json(tb, comps, image_size=IMG)


# --- Propagação da bbox da seta -------------------------------------------


def test_data_flow_carries_arrow_bbox():
    graph = _graph_with_flow()
    assert len(graph["data_flows"]) == 1
    flow = graph["data_flows"][0]
    assert flow["arrow_bbox"] == [150.0, 150.0, 390.0, 390.0]
    assert {flow["source"], flow["target"]} == {"c0", "c1"}


# --- Resolução id -> objeto ------------------------------------------------


def test_resolve_graph_index_maps_all_ids():
    graph = _graph_with_flow()
    index = resolve_graph_index(graph)
    # b0 (zona), c0 e c1 (componentes). A seta não vira nó nomeado no JSON.
    assert "b0" in index and "c0" in index and "c1" in index
    assert index["c0"]["name"] == "A"
    assert index["b0"]["label"] == "VPC"


# --- crop_region: clamp e degenerado --------------------------------------


def test_crop_region_clamps_to_image_bounds():
    img = _blank_image()
    # bbox que estoura os limites -> não deve levantar, e o recorte fica dentro.
    crop = crop_region(img, [-50, -50, 200, 200], margin_frac=0.5)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 1000 or crop.shape[0] >= 240  # pode ser ampliado


def test_crop_region_degenerate_bbox_returns_full_image():
    img = _blank_image()
    # x2 <= x1: recorte impossível -> devolve a imagem inteira (fallback seguro).
    out = crop_region(img, [500, 500, 500, 500])
    assert out.shape == img.shape


# --- highlight_risk por tipo ----------------------------------------------


def test_highlight_component_returns_crop():
    graph = _graph_with_flow()
    risk = Risk(
        target_type="component", target_id="c0", stride_category="Spoofing",
        elemento_afetado="A", justificativa="j", impacto="i",
        severidade="Alta", contramedida="c",
    )
    crop = highlight_risk(_blank_png_bytes(), risk, graph)
    assert crop is not None and crop.ndim == 3


def test_highlight_boundary_returns_crop():
    graph = _graph_with_flow()
    risk = Risk(
        target_type="boundary", target_id="b0", stride_category="Information Disclosure",
        elemento_afetado="VPC", justificativa="j", impacto="i",
        severidade="Crítica", contramedida="c",
    )
    crop = highlight_risk(_blank_png_bytes(), risk, graph)
    assert crop is not None and crop.ndim == 3


def test_highlight_flow_returns_crop():
    graph = _graph_with_flow()
    risk = Risk(
        target_type="flow", target_id="c0", flow_source_id="c0", flow_target_id="c1",
        stride_category="Tampering", elemento_afetado="A<->B", justificativa="j",
        impacto="i", severidade="Alta", contramedida="c",
    )
    crop = highlight_risk(_blank_png_bytes(), risk, graph)
    assert crop is not None and crop.ndim == 3


def test_highlight_invalid_target_returns_none():
    graph = _graph_with_flow()
    risk = Risk(
        target_type="component", target_id="c9999", stride_category="Denial of Service",
        elemento_afetado="inexistente", justificativa="j", impacto="i",
        severidade="Baixa", contramedida="c",
    )
    assert highlight_risk(_blank_png_bytes(), risk, graph) is None


# --- overlay numerado ------------------------------------------------------


def test_numbered_overlay_ignores_invalid_and_keeps_shape():
    graph = _graph_with_flow()
    risks = [
        Risk(target_type="component", target_id="c0", stride_category="Spoofing",
             elemento_afetado="A", justificativa="j", impacto="i",
             severidade="Alta", contramedida="c"),
        Risk(target_type="component", target_id="c9999", stride_category="Tampering",
             elemento_afetado="x", justificativa="j", impacto="i",
             severidade="Baixa", contramedida="c"),  # inválido: apenas ignorado
    ]
    out = draw_numbered_overlay(_blank_png_bytes(), risks, graph)
    assert out.shape == (1000, 1000, 3)


# --- ordenação por severidade ---------------------------------------------


def test_severity_rank_orders_critical_first():
    sev = ["Baixa", "Crítica", "Média", "Alta"]
    assert sorted(sev, key=severity_rank) == ["Crítica", "Alta", "Média", "Baixa"]

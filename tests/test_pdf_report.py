"""Testes da exportação em PDF (src/pdf_report.py).

Módulo puro (sem Streamlit): montamos groups/graph sintéticos e funções injetadas
falsas, geramos o PDF e o inspecionamos com pypdf — validando conteúdo real
(páginas, texto da capa, degradação sem imagem), não só "não quebrou".
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from pypdf import PdfReader

from src import theme
from src.pdf_report import _severity_summary, build_pdf_report
from src.stride_models import Risk, group_risks


def _risk(tt, tid, cat, sev, elem="Elemento") -> Risk:
    return Risk(
        target_type=tt, target_id=tid, flow_source_id=None, flow_target_id=None,
        stride_category=cat, elemento_afetado=elem, justificativa="justificativa",
        impacto="impacto", severidade=sev, contramedida="contramedida",
    )


def _overlay() -> np.ndarray:
    return np.full((300, 400, 3), 200, dtype=np.uint8)


def _card(_group, _graph) -> np.ndarray:
    return np.full((200, 300, 3), 180, dtype=np.uint8)


def _no_card(_group, _graph):
    return None


def _location(_group, _graph) -> str:
    return "VPC"


def _build(groups, *, card_fn=_card, name="diagrama.png") -> bytes:
    return build_pdf_report(
        diagram_name=name,
        groups=groups,
        graph={},  # as funções injetadas são falsas -> o grafo não é consultado
        numbered_overlay=_overlay(),
        element_location_fn=_location,
        card_image_fn=card_fn,
    )


def _text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() for page in reader.pages)


def _pages(pdf_bytes: bytes) -> int:
    return len(PdfReader(BytesIO(pdf_bytes)).pages)


# --- básico ----------------------------------------------------------------


def test_returns_pdf_bytes():
    groups = group_risks([_risk("component", "c0", "Spoofing", "Alta")])
    pdf = _build(groups)
    assert isinstance(pdf, bytes) and pdf[:4] == b"%PDF"


def test_opens_and_has_cover_plus_block():
    groups = group_risks([_risk("component", "c0", "Spoofing", "Alta")])
    # capa + mapa de riscos + 1 bloco -> >= 2 páginas.
    assert _pages(_build(groups)) >= 2


# --- conteúdo da capa ------------------------------------------------------


def test_cover_contains_diagram_name_and_severity_counts():
    groups = group_risks([
        _risk("component", "c0", "Spoofing", "Crítica"),
        _risk("component", "c1", "Tampering", "Alta"),
        _risk("component", "c2", "Repudiation", "Alta"),
    ])
    text = _text(_build(groups, name="minha_arquitetura.png"))
    assert "minha_arquitetura.png" in text
    assert "Parecer STRIDE" in text
    # 1 Crítica · 2 Altas — o resumo aparece na capa.
    assert "1 Crítica" in text
    assert "2 Altas" in text


def test_severity_summary_orders_and_pluralizes():
    groups = group_risks([
        _risk("component", "c0", "Spoofing", "Crítica"),
        _risk("component", "c1", "Tampering", "Baixa"),
        _risk("component", "c2", "Repudiation", "Baixa"),
    ])
    # pior primeiro; plural em "Baixas", singular em "Crítica".
    assert _severity_summary(groups) == "1 Crítica · 2 Baixas"


# --- escala com nº de grupos ----------------------------------------------


def test_more_groups_yield_more_pages():
    one = group_risks([_risk("component", "c0", "Spoofing", "Alta")])
    many = group_risks([
        _risk("component", f"c{i}", "Spoofing", "Alta", elem=f"E{i}") for i in range(5)
    ])
    assert _pages(_build(many)) > _pages(_build(one))


# --- degradação sem imagem -------------------------------------------------


def test_missing_card_image_does_not_break_and_keeps_text():
    groups = group_risks([_risk("component", "c0", "Spoofing", "Alta", elem="Gateway")])
    pdf = _build(groups, card_fn=_no_card)
    assert pdf[:4] == b"%PDF"
    text = _text(pdf)
    assert "Gateway" in text
    assert "sem localiza" in text.lower()  # placeholder de "sem localização visual"


# --- severidade aparece no bloco -------------------------------------------


def test_block_text_contains_stride_fields():
    groups = group_risks([
        _risk("component", "c0", "Denial of Service", "Crítica", elem="Load Balancer"),
    ])
    text = _text(_build(groups))
    assert "Load Balancer" in text
    assert "Denial of Service" in text
    assert "Crítica" in text
    assert "contramedida" in text

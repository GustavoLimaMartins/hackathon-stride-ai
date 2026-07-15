"""Exportação do parecer STRIDE em PDF (download).

Monta um PDF a partir dos MESMOS dados já calculados na sessão do Streamlit
(o grafo, os grupos de risco consolidados, o overlay numerado) — não há
recálculo do pipeline, apenas uma nova camada de renderização. O documento segue
a identidade visual "FIAP Software Security" (magenta de marca, faixas de
severidade), lendo os hex de src/theme.py para não divergir da tela.

O PDF usa um tema CLARO/editorial (fundo branco, cabeçalhos escuros, accent
magenta) — não o dark mode 1:1 da tela — porque texto claro sobre fundo preto
consome tinta e prejudica a leitura impressa. É a mesma orientação do guia de
marca de tratar dark/light como temas distintos.

Biblioteca: ReportLab (puro Python, sem binários externos). Desenha diretamente
a partir dos arrays numpy/RGB que visual_report.py já produz.

Módulo puro (sem Streamlit): recebe tudo por parâmetro e devolve os bytes do PDF,
para ser testável isoladamente e para main.py apenas injetar os dados e passar o
resultado ao st.download_button.
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import theme
from src.stride_models import RiskGroup, severity_rank

# --- Cores do documento (tema claro editorial, lendo a marca de theme.py) ---
_C_INK = colors.HexColor(theme.INK)
_C_NAVY = colors.HexColor(theme.NAVY)
_C_MAGENTA = colors.HexColor(theme.MAGENTA)
_C_TEXT_ON_DARK = colors.HexColor(theme.TEXT_HI)
_C_PAGE_TEXT = colors.HexColor("#1A1A22")   # texto do corpo (escuro sobre branco)
_C_MUTED = colors.HexColor("#5A5766")       # rótulos secundários
_C_LINE = colors.HexColor("#DDDAE2")        # linhas de tabela discretas
_C_ROW_ALT = colors.HexColor("#F5F3F7")     # zebra sutil nas linhas da tabela

# Nome do produto (rodapé + capa).
_BRAND = "FIAP Software Security"
_PRODUCT = "STRIDE-AI"

# Layout de página.
_PAGE = A4
_MARGIN = 18 * mm
_CONTENT_W = _PAGE[0] - 2 * _MARGIN


def _severity_color(severidade: str) -> colors.Color:
    """Cor ReportLab da severidade (via a escala semântica de theme.py)."""
    return colors.HexColor(theme.severity_color(severidade))


def _np_to_flowable(img: np.ndarray, max_w: float, max_h: float) -> RLImage:
    """Array RGB -> Image flowable do ReportLab, encaixado em (max_w, max_h).

    Converte o array para PNG em memória (via Pillow, já disponível no ambiente),
    embrulha num ImageReader e escala preservando a proporção para caber na caixa
    dada — a mesma filosofia de letterbox usada nos cards da tela, aqui só para
    não estourar a largura/altura da página.
    """
    buf = BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    buf.seek(0)
    reader = ImageReader(buf)
    iw, ih = reader.getSize()
    scale = min(max_w / iw, max_h / ih)
    return RLImage(buf, width=iw * scale, height=ih * scale)


def _severity_summary(groups: list[RiskGroup]) -> str:
    """'2 Críticas · 5 Altas · 3 Médias' — contagem por severidade, pior primeiro.

    Conta TODOS os riscos (não só os representantes), na ordem canônica de
    SEVERITY_ORDER, omitindo severidades sem ocorrência.
    """
    counts: Counter[str] = Counter()
    for group in groups:
        for risk in group.risks:
            counts[risk.severidade] += 1
    parts = [
        f"{counts[sev]} {sev}{'s' if counts[sev] > 1 else ''}"
        for sev in sorted(counts, key=severity_rank)
    ]
    return " · ".join(parts) if parts else "Nenhum risco identificado"


def _styles() -> dict[str, ParagraphStyle]:
    """Estilos de parágrafo do documento (título, corpo, célula de tabela)."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "STRIDETitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=26, leading=30, textColor=_C_PAGE_TEXT, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "STRIDESubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=15, textColor=_C_MUTED,
        ),
        "h2": ParagraphStyle(
            "STRIDEH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, textColor=_C_PAGE_TEXT, spaceBefore=2, spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "STRIDEMeta", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13, textColor=_C_MUTED,
        ),
        "cell": ParagraphStyle(
            "STRIDECell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=_C_PAGE_TEXT, alignment=TA_LEFT,
        ),
        "cell_head": ParagraphStyle(
            "STRIDECellHead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=11, textColor=_C_TEXT_ON_DARK, alignment=TA_LEFT,
        ),
    }


def _cover(story: list, styles: dict, diagram_name: str, groups: list[RiskGroup]) -> None:
    """Bloco de capa: título, diagrama analisado, resumo por severidade, selo."""
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Parecer STRIDE", styles["title"]))
    story.append(Paragraph(
        "Análise de ameaças em diagrama de arquitetura cloud", styles["subtitle"],
    ))
    story.append(Spacer(1, 10 * mm))

    # Faixa de marca (retângulo magenta com o nome da empresa).
    brand = Table(
        [[Paragraph(f"<b>{_BRAND}</b>  ·  {_PRODUCT}", ParagraphStyle(
            "brand", fontName="Helvetica-Bold", fontSize=12,
            textColor=_C_TEXT_ON_DARK, leading=16))]],
        colWidths=[_CONTENT_W],
    )
    brand.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _C_MAGENTA),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(brand)
    story.append(Spacer(1, 8 * mm))

    meta = Table(
        [
            [Paragraph("<b>Diagrama analisado</b>", styles["meta"]),
             Paragraph(diagram_name, styles["meta"])],
            [Paragraph("<b>Elementos com risco</b>", styles["meta"]),
             Paragraph(str(len(groups)), styles["meta"])],
            [Paragraph("<b>Riscos por severidade</b>", styles["meta"]),
             Paragraph(_severity_summary(groups), styles["meta"])],
        ],
        colWidths=[45 * mm, _CONTENT_W - 45 * mm],
    )
    meta.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, _C_LINE),
    ]))
    story.append(meta)


def _risk_map(story: list, styles: dict, numbered_overlay: np.ndarray) -> None:
    """Página do mapa de riscos: o overlay numerado como imagem grande."""
    story.append(PageBreak())
    story.append(Paragraph("Mapa de riscos", styles["h2"]))
    story.append(Paragraph(
        "Cada número marca o elemento do bloco correspondente nas páginas "
        "seguintes.", styles["meta"],
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(_np_to_flowable(numbered_overlay, _CONTENT_W, 210 * mm))


def _risk_table(group: RiskGroup, styles: dict) -> Table:
    """Tabela consolidada do grupo: 4 colunas, uma linha por ameaça do elemento.

    Cabeçalho em navy; cada linha ganha uma faixa lateral (borda esquerda) na cor
    da severidade daquela linha — a mesma hierarquia visual por severidade usada
    na tela. Zebra sutil para legibilidade em tabelas longas.
    """
    header = [
        Paragraph("Categoria STRIDE", styles["cell_head"]),
        Paragraph("Justificativa", styles["cell_head"]),
        Paragraph("Severidade", styles["cell_head"]),
        Paragraph("Contramedida", styles["cell_head"]),
    ]
    rows = [header]
    for risk in group.risks:
        rows.append([
            Paragraph(risk.stride_category, styles["cell"]),
            Paragraph(risk.justificativa, styles["cell"]),
            Paragraph(risk.severidade, styles["cell"]),
            Paragraph(risk.contramedida, styles["cell"]),
        ])

    col_w = [30 * mm, _CONTENT_W - 30 * mm - 22 * mm - 45 * mm, 22 * mm, 45 * mm]
    table = Table(rows, colWidths=col_w, repeatRows=1)

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), _C_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _C_LINE),
        ("GRID", (0, 0), (-1, -1), 0.25, _C_LINE),
    ]
    # Faixa lateral colorida por linha (borda esquerda grossa na cor da severidade)
    # + zebra sutil nas linhas de dados.
    for i, risk in enumerate(group.risks, start=1):
        style.append(("LINEBEFORE", (0, i), (0, i), 3, _severity_color(risk.severidade)))
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), _C_ROW_ALT))
    table.setStyle(TableStyle(style))
    return table


def _risk_block(
    story: list,
    styles: dict,
    n: int,
    group: RiskGroup,
    graph: dict,
    location: str,
    card: np.ndarray | None,
) -> None:
    """Um bloco por elemento: cabeçalho + imagem (ou placeholder) + tabela."""
    rep = group.representative
    story.append(PageBreak())

    # Cabeçalho "#N — elemento · severidade" com faixa lateral de severidade.
    head = Table(
        [[Paragraph(
            f"<b>#{n} — {rep.elemento_afetado}</b>  ·  {rep.severidade}",
            ParagraphStyle("blockhead", fontName="Helvetica-Bold", fontSize=13,
                           textColor=_C_PAGE_TEXT, leading=17))]],
        colWidths=[_CONTENT_W],
    )
    head.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, 0), 5, _severity_color(rep.severidade)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(head)
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"<b>Elemento:</b> {rep.elemento_afetado} &nbsp;&nbsp; "
        f"<b>Localização:</b> {location}", styles["meta"],
    ))
    story.append(Spacer(1, 4 * mm))

    if card is not None:
        story.append(_np_to_flowable(card, _CONTENT_W, 95 * mm))
    else:
        story.append(Paragraph(
            f"<i>Sem localização visual disponível para "
            f"{rep.elemento_afetado}.</i>", styles["meta"],
        ))
    story.append(Spacer(1, 4 * mm))
    story.append(_risk_table(group, styles))


def _draw_footer(canvas, doc) -> None:
    """Rodapé em todas as páginas: produto à esquerda, paginação à direita."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_C_MUTED)
    y = 10 * mm
    canvas.drawString(_MARGIN, y, f"{_BRAND} · {_PRODUCT}")
    canvas.drawRightString(_PAGE[0] - _MARGIN, y, f"Página {doc.page}")
    canvas.setStrokeColor(_C_LINE)
    canvas.setLineWidth(0.5)
    canvas.line(_MARGIN, y + 4 * mm, _PAGE[0] - _MARGIN, y + 4 * mm)
    canvas.restoreState()


def build_pdf_report(
    *,
    diagram_name: str,
    groups: list[RiskGroup],
    graph: dict,
    numbered_overlay: np.ndarray,
    element_location_fn: Callable[[RiskGroup, dict], str],
    card_image_fn: Callable[[RiskGroup, dict], np.ndarray | None],
) -> bytes:
    """Monta o parecer STRIDE em PDF e devolve os bytes (para st.download_button).

    Recebe os mesmos dados já calculados na sessão:
      - diagram_name: nome do arquivo do diagrama enviado (aparece na capa/nome);
      - groups: os RiskGroup consolidados (já ordenados por pior severidade);
      - graph: o grafo hierárquico (para as funções injetadas resolverem ids);
      - numbered_overlay: o array RGB do overlay numerado (draw_numbered_overlay);
      - element_location_fn(group, graph) -> str: zona/localização do elemento;
      - card_image_fn(group, graph) -> np.ndarray|None: recorte padronizado do
        elemento (highlight_element), ou None se não localizável.

    As duas funções são injetadas para reaproveitar a lógica de main.py/
    visual_report.py sem duplicá-la e sem este módulo conhecer o UploadedFile.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=_PAGE,
        leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_MARGIN, bottomMargin=_MARGIN + 6 * mm,
        title=f"Parecer STRIDE — {diagram_name}", author=_BRAND,
    )
    styles = _styles()
    story: list = []

    _cover(story, styles, diagram_name, groups)
    _risk_map(story, styles, numbered_overlay)
    for n, group in enumerate(groups, start=1):
        location = element_location_fn(group, graph)
        card = card_image_fn(group, graph)
        _risk_block(story, styles, n, group, graph, location, card)

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buf.getvalue()

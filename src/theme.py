"""Fonte única da paleta de marca "FIAP Software Security".

Espelha 1:1 os tokens do guia de identidade visual (o Artifact aprovado): preto
(base), magenta (marca) e navy (superfície auxiliar), mais as quatro cores
semânticas de severidade STRIDE. Tanto o front-end (visual_report/main) quanto o
back-end de exportação (pdf_report) importam daqui, para o hex nunca divergir
entre o que a tela mostra e o que o PDF baixado mostra.

Módulo puro (sem Streamlit, sem ReportLab, sem OpenCV) para ser importável de
qualquer camada.
"""

from __future__ import annotations

# --- Paleta de marca (tokens do guia de design) ---------------------------
INK = "#050507"          # base neutra: fundo/superfície principal do dark mode
NAVY = "#12183A"         # superfície auxiliar: cards, sidebar, painéis
MAGENTA = "#EC0868"      # cor de marca: logo, ações, destaques de risco
MAGENTA_HI = "#FF2F85"   # variação clara do magenta (hover/realce)
TEXT_HI = "#F7F5F3"      # texto de alto contraste sobre fundo escuro

# --- Severidade STRIDE (escala semântica, independente do accent de marca) --
# As chaves batem com stride_models.SEVERITY_ORDER.
SEVERITY_COLORS: dict[str, str] = {
    "Crítica": "#FF5A52",
    "Alta": "#FFAB3D",
    "Média": "#4D9DFF",
    "Baixa": "#6FAE8A",
}

# Cor neutra para severidade desconhecida (defensivo — nunca deve acontecer, pois
# o schema restringe o campo a um Literal, mas evita KeyError se algo mudar).
_SEVERITY_FALLBACK = "#7C7891"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (r, g, b). Aceita com ou sem o '#' inicial."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def severity_color(severidade: str) -> str:
    """Hex da cor semântica de uma severidade (fallback neutro se desconhecida)."""
    return SEVERITY_COLORS.get(severidade, _SEVERITY_FALLBACK)


def severity_color_rgb(severidade: str) -> tuple[int, int, int]:
    """Cor de severidade como tupla RGB (para OpenCV/ReportLab)."""
    return hex_to_rgb(severity_color(severidade))

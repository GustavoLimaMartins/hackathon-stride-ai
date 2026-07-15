"""Camada de apresentação da marca "FIAP Software Security" no Streamlit.

O Streamlit não expõe um HTML estático substituível — o DOM é gerado via React/
emotion. A forma padrão e de baixo risco de aplicar a identidade visual do guia
(o Artifact aprovado) é injetar um bloco de CSS que mira os atributos
`[data-testid="..."]` (mais estáveis que as classes auto-geradas) e desenhar o
chrome estático (hero, rodapé) como HTML via st.markdown(unsafe_allow_html=True).

Escopo: recolore/restila widgets nativos e substitui o cabeçalho/rodapé genéricos
pela identidade da marca. NÃO reconstrói widgets interativos (file_uploader,
progress) — esses só recebem cor, pois um substituto via st.components.v1.html
roda em iframe sandboxed e não teria acesso ao arquivo enviado.

Os tokens de cor completos do guia vivem AQUI como CSS custom properties (só-de-
apresentação); theme.py permanece enxuto (constantes consumidas por lógica real:
PDF, highlight de risco). Os hex que aparecem nos dois lugares são a MESMA fonte
visual (o guia) em dois formatos de consumo — Python vs CSS.
"""

from __future__ import annotations

import streamlit as st

from src import theme

# Mapeia cada severidade para a classe da pill do guia (sev-critical-pill etc.).
_SEVERITY_PILL_CLASS: dict[str, str] = {
    "Crítica": "sev-critical-pill",
    "Alta": "sev-high-pill",
    "Média": "sev-medium-pill",
    "Baixa": "sev-low-pill",
}

# Slogan de assinatura e selo combinado (seções 1 e 2 do guia).
_SLOGAN = "Todo diagrama esconde um ponto de falha. Nós o desenhamos de volta."
_COMBINED_MARK = (
    "<span class='product'>STRIDE-AI</span> <span class='by'>by</span> "
    "<span class='company'>FIAP Software Security</span>"
)

# Bloco de CSS único: @import de fontes, token system do guia, ocultar chrome
# default e reskin dos widgets nativos. Injetado uma vez, no topo do main.py.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Roboto:wght@300;400;500;700&family=JetBrains+Mono:wght@400&display=swap');

:root {
  --ink: #050507; --ink-1: #0a0a0f; --ink-2: #101018;
  --navy: #12183a; --navy-1: #171f47; --navy-2: #202a5c;
  --magenta: #ec0868; --magenta-1: #ff2f85; --magenta-dim: #7a0c3d;
  --paper: #f5f3f1;
  --line: rgba(245,243,241,0.12); --line-strong: rgba(245,243,241,0.22);
  --text-hi: #f7f5f3; --text-mid: #c7c3d4; --text-low: #7c7891;
  --sev-critical: #ff5a52; --sev-high: #ffab3d; --sev-medium: #4d9dff;
  --sev-low: #6fae8a; --ok: #3ecf8e;
  --font-display: 'Space Grotesk', 'Arial Narrow', sans-serif;
  --font-body: 'Roboto', -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
}

/* --- Tipografia global --- */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: var(--font-body);
}
h1, h2, h3, .brand-mark, .brand-eyebrow {
  font-family: var(--font-display) !important;
  letter-spacing: -0.01em;
}
code, pre, [data-testid="stJson"], .mono { font-family: var(--font-mono) !important; }

/* --- Ocultar chrome default do Streamlit --- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

/* --- Container / espaçamento --- */
.block-container { max-width: 1160px; padding-top: 2.5rem; }

/* --- Botões (primary + download) --- */
.stButton > button,
[data-testid="stDownloadButton"] > button {
  background: var(--magenta);
  color: #fff;
  border: 1px solid transparent;
  border-radius: 8px;
  font-weight: 500;
  transition: background 0.15s;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
  background: var(--magenta-1);
  color: #fff;
}
.stButton > button:focus-visible,
[data-testid="stDownloadButton"] > button:focus-visible {
  outline: 2px solid var(--magenta-1);
  outline-offset: 2px;
}

/* --- File uploader: apenas recolorido (estrutura React intacta) --- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--navy);
  border: 1.5px dashed var(--line-strong);
  border-radius: 12px;
}

/* --- Barra de progresso: preenchimento com gradiente do guia --- */
[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, var(--navy-2), var(--magenta-1));
  border-radius: 999px;
}
[data-testid="stProgress"] > div > div {
  background: var(--navy);
  border-radius: 999px;
}

/* --- Alerts (info/error) --- */
[data-testid="stAlert"] {
  border-radius: 10px;
  border-left: 3px solid var(--magenta);
}

/* --- Expander --- */
[data-testid="stExpander"] summary {
  background: var(--navy);
  border-radius: 8px;
}

/* --- Card do bloco de risco (st.container(border=True)) --- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--navy-1);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.25rem 0.5rem;
}
[data-testid="stImage"] img { border-radius: 8px; }

/* --- Tabela STRIDE consolidada (_risk_table_html) --- */
.risk-table table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.risk-table th {
  text-align: left;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-low);
  font-weight: 500;
  padding: 0 10px 8px 0;
}
.risk-table td {
  padding: 8px 10px 8px 0;
  border-top: 1px solid var(--line);
  vertical-align: top;
  color: var(--text-mid);
}
.risk-table td:first-child { color: var(--text-hi); font-weight: 500; }

/* --- Pill de severidade (seções 3/9 do guia) --- */
.sev-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 4px 12px 4px 9px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 500;
  white-space: nowrap;
}
.sev-pill i {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 25%, transparent);
}
.sev-critical-pill { background: color-mix(in srgb, var(--sev-critical) 16%, var(--navy)); color: var(--sev-critical); }
.sev-high-pill { background: color-mix(in srgb, var(--sev-high) 16%, var(--navy)); color: var(--sev-high); }
.sev-medium-pill { background: color-mix(in srgb, var(--sev-medium) 16%, var(--navy)); color: var(--sev-medium); }
.sev-low-pill { background: color-mix(in srgb, var(--sev-low) 16%, var(--navy)); color: var(--sev-low); }

/* --- Hero de marca --- */
.brand-hero {
  padding: 8px 0 20px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 8px;
}
.brand-eyebrow {
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--magenta-1);
  margin-bottom: 10px;
}
.brand-mark {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: clamp(2.2rem, 5vw, 3.4rem);
  line-height: 1.0;
  color: var(--text-hi);
}
.brand-mark .brand-mark-accent { color: var(--magenta-1); }
.brand-tagline {
  margin: 18px 0 0;
  font-size: 1.35rem;
  font-weight: 400;
  color: var(--text-hi);
  max-width: 62ch;
}

/* --- Rodapé de marca --- */
.brand-footer {
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--line);
  color: var(--text-low);
}
.brand-footer .slogan {
  font-family: var(--font-display);
  font-weight: 500;
  font-size: 1.15rem;
  color: var(--text-mid);
  margin-bottom: 16px;
  max-width: 68ch;
  text-wrap: balance;
}
.brand-footer .slogan .hl { color: var(--magenta-1); }
.brand-footer .mark {
  font-family: var(--font-display);
  letter-spacing: 0.01em;
  color: var(--text-hi);
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
}
.brand-footer .mark .product {
  font-weight: 500;
  font-size: 0.95rem;
  color: var(--text-mid);
}
.brand-footer .mark .by {
  color: var(--text-low);
  font-weight: 300;
  font-size: 0.85rem;
}
.brand-footer .mark .company {
  font-weight: 700;
  font-size: 1.5rem;
  color: var(--text-hi);
}
</style>
"""


def inject_brand_css() -> None:
    """Injeta o CSS da marca. Chamar uma vez, logo após set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    """Cabeçalho de marca: empresa (eyebrow) + produto STRIDE·AI + tagline.

    Substitui st.title(); a hierarquia segue a seção 1 do guia (empresa como selo
    de origem acima, produto como nome grande que o usuário opera).
    """
    st.markdown(
        """
        <div class="brand-hero">
          <div class="brand-eyebrow">FIAP Software Security</div>
          <div class="brand-mark">STRIDE<span class="brand-mark-accent">·AI</span></div>
          <p class="brand-tagline">Análise de ameaças STRIDE em diagramas de arquitetura cloud</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Rodapé de marca no lugar do 'Made with Streamlit' oculto.

    Traz o slogan de assinatura (seção 2) e o selo combinado 'STRIDE-AI by FIAP
    Software Security' (seção 1) — a marca readquire presença que o rodapé default
    oculto deixaria vazia.
    """
    slogan = _SLOGAN.replace(
        "um ponto de falha", "<span class='hl'>um ponto de falha</span>"
    )
    st.markdown(
        f"""
        <div class="brand-footer">
          <div class="slogan">{slogan}</div>
          <div class="mark">{_COMBINED_MARK}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_pill_html(severidade: str) -> str:
    """Pill de severidade do guia: '<span class=sev-pill sev-...><i></i>Sev</span>'.

    A classe determina a cor semântica (via CSS injetado); o texto é a própria
    severidade. Severidade fora do mapa cai numa pill neutra (sem classe de cor).
    """
    pill_class = _SEVERITY_PILL_CLASS.get(severidade, "")
    return (
        f"<span class='sev-pill {pill_class}'><i></i>{severidade}</span>"
    )

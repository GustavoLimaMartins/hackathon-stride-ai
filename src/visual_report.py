"""Renderização visual: desenha as detecções do YOLO sobre a imagem do diagrama.

Complementa o parecer textual do LLM com uma visão espacial de "o que o modelo
viu" — cada componente e cada zona de confiança detectados ganham uma caixa
colorida por classe, exibida no Streamlit ao lado da análise STRIDE.

O desenho vive aqui (não em vision.py, que só faz inferência) para manter a
separação de responsabilidades já usada no projeto.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from src.vision import _load_image

# Cor (R, G, B) por classe. A imagem é tratada em RGB (mesmo espaço que o
# pipeline usa), então as tuplas aqui são RGB — não BGR.
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "trust_boundary": (33, 118, 255),   # azul
    "compute": (255, 140, 0),           # laranja
    "database_storage": (46, 160, 67),  # verde
    "api_gateway": (163, 73, 220),      # roxo
    "network_security": (220, 38, 38),  # vermelho
    "actor": (240, 200, 20),            # amarelo
    "data_flow": (0, 190, 200),         # ciano
}

_DEFAULT_COLOR = (128, 128, 128)  # cinza, para classes fora do mapa (defensivo)

# Cor das linhas de proximity_hint (cinza claro): sinal secundário, não deve
# competir visualmente com as bboxes coloridas das classes.
_PROXIMITY_COLOR = (150, 150, 150)


def class_color(class_name: str) -> tuple[int, int, int]:
    """Cor RGB associada a uma classe (cinza se a classe for desconhecida)."""
    return _CLASS_COLORS.get(class_name, _DEFAULT_COLOR)


def _draw_box(
    img: np.ndarray,
    bbox: list,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
) -> None:
    """Desenha um retângulo + rótulo (com fundo) sobre a imagem, in-place."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, text_thickness)

    # Faixa de fundo atrás do texto, para legibilidade sobre o diagrama.
    top = max(0, y1 - th - baseline - 2)
    cv2.rectangle(img, (x1, top), (x1 + tw + 2, y1), color, -1)
    # Texto em branco ou preto conforme o brilho da cor de fundo (contraste).
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    cv2.putText(
        img,
        label,
        (x1 + 1, y1 - baseline - 1),
        font,
        scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 12,
) -> None:
    """Desenha uma linha TRACEJADA entre dois pontos, in-place.

    O OpenCV não tem linha tracejada nativa, então percorremos o segmento em
    passos de 'dash' px e desenhamos só os trechos alternados. O tracejado
    distingue visualmente um proximity_hint (relação inferida, mais fraca) de
    uma bbox sólida — e de um eventual fluxo sólido no futuro.
    """
    x1, y1 = p1
    x2, y2 = p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length == 0:
        return
    steps = max(int(length / dash), 1)
    for k in range(steps):
        if k % 2 == 1:  # pula os trechos ímpares -> tracejado
            continue
        a = k / steps
        b = min((k + 1) / steps, 1.0)
        xa, ya = int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)
        xb, yb = int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)
        cv2.line(img, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def draw_overlay(
    image: str | Path | bytes | BytesIO,
    trust_boundaries: list[dict],
    components: list[dict],
    proximity_lines: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> np.ndarray:
    """Desenha as bounding boxes das detecções sobre a imagem enviada.

    Reaproveita vision._load_image() para abrir exatamente a mesma imagem que
    gerou as bboxes. Cada zona de confiança (trust_boundary) é desenhada com
    traço mais grosso e rotulada com o nome lido pelo OCR ('label') quando
    disponível; os demais componentes são rotulados com 'classe confiança'.

    'proximity_lines' (opcional) são pares de pontos (centróides) a ligar com
    uma linha tracejada cinza — a representação visual dos proximity_hints do
    grafo (relações inferidas por proximidade). Sem elas, o overlay é idêntico
    ao comportamento anterior.

    Retorna um array Numpy RGB pronto para st.image().
    """
    img = np.array(_load_image(image).convert("RGB"))

    # Linhas de proximidade primeiro (por baixo das bboxes, para não obscurecê-las).
    for p1, p2 in proximity_lines or []:
        _draw_dashed_line(img, p1, p2, _PROXIMITY_COLOR, thickness=2)

    # Zonas primeiro (traço grosso), para os componentes ficarem por cima.
    for boundary in trust_boundaries:
        label = boundary.get("label") or "trust_boundary"
        _draw_box(
            img,
            boundary["bbox"],
            class_color("trust_boundary"),
            label,
            thickness=3,
        )

    for component in components:
        cls = component["class"]
        label = f"{cls} {component['confidence']:.2f}"
        _draw_box(img, component["bbox"], class_color(cls), label, thickness=2)

    return img


def legend_items(
    trust_boundaries: list[dict], components: list[dict]
) -> list[tuple[str, str]]:
    """Legenda (nome_da_classe, cor_hex) só das classes de fato presentes.

    Evita poluir a legenda com classes que não aparecem no diagrama atual.
    Retorna pares prontos para o main.py montar a legenda visual.
    """
    present = {"trust_boundary"} if trust_boundaries else set()
    present.update(c["class"] for c in components)

    items: list[tuple[str, str]] = []
    for cls in _CLASS_COLORS:  # ordem estável do dict
        if cls in present:
            r, g, b = class_color(cls)
            items.append((cls, f"#{r:02x}{g:02x}{b:02x}"))
    return items

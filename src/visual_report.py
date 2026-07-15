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

# Cor do DESTAQUE de um risco (vermelho forte): a caixa do elemento afetado por
# um risco recebe esta cor, sobressaindo às cores neutras de classe no recorte.
_RISK_HIGHLIGHT_COLOR = (255, 0, 0)
# Cor do contexto secundário num recorte de risco (ex.: origem/destino de um
# fluxo, ou componentes internos de uma zona) — traço fino e discreto.
_RISK_CONTEXT_COLOR = (90, 90, 90)


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


# ---------------------------------------------------------------------------
# Rastreabilidade visual de riscos: recorte + destaque do elemento afetado.
# ---------------------------------------------------------------------------

# Altura mínima (px) de um recorte de risco; recortes menores são ampliados para
# ficarem legíveis na UI (mesmo racional de ocr_engine._upscale_if_small).
_MIN_RISK_CROP_HEIGHT = 240


def resolve_graph_index(graph: dict) -> dict[str, dict]:
    """Mapa id -> objeto do grafo, para localizar bboxes por id (cN, bN).

    Varre zonas (bN), componentes aninhados e unassigned (cN), reunindo cada
    objeto sob o seu id. É a base para localizar o alvo de um risco a partir do
    target_id devolvido pelo LLM. Reúne o padrão de varredura antes espalhado
    (ver main._proximity_lines), agora que UI e destaque de risco precisam dele.
    """
    index: dict[str, dict] = {}
    for boundary in graph.get("trust_boundaries", []):
        index[boundary["id"]] = boundary
        for comp in boundary.get("components", []):
            index[comp["id"]] = comp
    for comp in graph.get("unassigned_components", []):
        index[comp["id"]] = comp
    return index


def _find_data_flow(graph: dict, id_a: str | None, id_b: str | None) -> dict | None:
    """Localiza o data_flow que liga o par {id_a, id_b} (não-ordenado)."""
    if not id_a or not id_b:
        return None
    want = {id_a, id_b}
    for flow in graph.get("data_flows", []):
        if {flow["source"], flow["target"]} == want:
            return flow
    return None


def _clamp_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    """bbox -> inteiros dentro dos limites da imagem (padrão de ocr_engine)."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return x1, y1, x2, y2


def crop_region(img: np.ndarray, bbox, margin_frac: float = 0.15) -> np.ndarray:
    """Recorta a região de 'bbox' com uma margem ao redor, ampliando se pequena.

    A margem (fração do lado do bbox) dá contexto ao redor do elemento. Os
    limites são clampados à imagem; recortes baixos são ampliados para
    _MIN_RISK_CROP_HEIGHT. Retorna um array RGB pronto para st.image.
    """
    height, width = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    mx = (x2 - x1) * margin_frac
    my = (y2 - y1) * margin_frac
    cx1, cy1, cx2, cy2 = _clamp_bbox((x1 - mx, y1 - my, x2 + mx, y2 + my), width, height)
    if cx2 <= cx1 or cy2 <= cy1:
        return img  # bbox degenerado: devolve a imagem inteira (fallback seguro)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.shape[0] < _MIN_RISK_CROP_HEIGHT:
        scale = _MIN_RISK_CROP_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _envelope(*bboxes) -> list[float]:
    """Menor retângulo que contém todos os bboxes dados."""
    xs1 = [b[0] for b in bboxes]
    ys1 = [b[1] for b in bboxes]
    xs2 = [b[2] for b in bboxes]
    ys2 = [b[3] for b in bboxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def highlight_risk(
    image: str | Path | bytes | BytesIO, risk, graph: dict
) -> np.ndarray | None:
    """Imagem contextual de um risco: destaca o elemento afetado e recorta a região.

    'risk' é um objeto com os campos target_type ('component'|'flow'|'boundary'),
    target_id e, para fluxos, flow_source_id/flow_target_id. Conforme o tipo:
      - component: destaca a bbox do componente e recorta ao redor dele;
      - flow: destaca a seta (arrow_bbox do data_flow) e as duas pontas como
        contexto; recorta no envelope dos três;
      - boundary: destaca a zona e seus componentes internos (contexto), recorta
        na zona.
    Retorna o recorte RGB, ou None se o target_id não existir no grafo (a UI
    então mostra o card sem imagem — degradação elegante).
    """
    index = resolve_graph_index(graph)
    img = np.array(_load_image(image).convert("RGB"))

    ttype = getattr(risk, "target_type", None)

    if ttype == "boundary":
        zone = index.get(risk.target_id)
        if zone is None:
            return None
        for comp in zone.get("components", []):  # contexto: filhos da zona
            _draw_box(img, comp["bbox"], _RISK_CONTEXT_COLOR, "", thickness=2)
        _draw_box(img, zone["bbox"], _RISK_HIGHLIGHT_COLOR, "", thickness=4)
        return crop_region(img, zone["bbox"], margin_frac=0.05)

    if ttype == "flow":
        src = index.get(risk.flow_source_id)
        tgt = index.get(risk.flow_target_id)
        flow = _find_data_flow(graph, risk.flow_source_id, risk.flow_target_id)
        boxes = [o["bbox"] for o in (src, tgt) if o is not None]
        if flow is not None:
            boxes.append(flow["arrow_bbox"])
        if not boxes:
            return None
        # Contexto: origem e destino em traço fino.
        for o in (src, tgt):
            if o is not None:
                _draw_box(img, o["bbox"], _RISK_CONTEXT_COLOR, "", thickness=2)
        # Destaque: a seta, quando existe; senão, o envelope das pontas.
        highlight_box = flow["arrow_bbox"] if flow is not None else _envelope(*boxes)
        _draw_box(img, highlight_box, _RISK_HIGHLIGHT_COLOR, "", thickness=4)
        return crop_region(img, _envelope(*boxes), margin_frac=0.2)

    # component (default)
    comp = index.get(risk.target_id)
    if comp is None:
        return None
    _draw_box(img, comp["bbox"], _RISK_HIGHLIGHT_COLOR, "", thickness=4)
    return crop_region(img, comp["bbox"], margin_frac=0.25)


def _risk_anchor_bbox(risk, graph: dict, index: dict[str, dict]):
    """bbox onde ancorar o número (#N) de um risco no overlay global, ou None."""
    ttype = getattr(risk, "target_type", None)
    if ttype == "flow":
        flow = _find_data_flow(graph, risk.flow_source_id, risk.flow_target_id)
        if flow is not None:
            return flow["arrow_bbox"]
        src = index.get(risk.flow_source_id)
        tgt = index.get(risk.flow_target_id)
        boxes = [o["bbox"] for o in (src, tgt) if o is not None]
        return _envelope(*boxes) if boxes else None
    obj = index.get(risk.target_id)
    return obj["bbox"] if obj is not None else None


def draw_numbered_overlay(
    image: str | Path | bytes | BytesIO, risks, graph: dict
) -> np.ndarray:
    """Overlay global com o bbox de cada risco marcado pelo seu número (#N).

    Cada risco em 'risks' (na ordem já exibida na UI) é desenhado sobre a imagem
    com a caixa do elemento afetado e o rótulo "#<posição>", dando a visão de
    conjunto de onde estão todos os pontos que precisam de intervenção. Riscos
    cujo alvo não existe no grafo são ignorados (não têm onde ancorar).
    """
    img = np.array(_load_image(image).convert("RGB"))
    index = resolve_graph_index(graph)
    for i, risk in enumerate(risks, start=1):
        bbox = _risk_anchor_bbox(risk, graph, index)
        if bbox is not None:
            _draw_box(img, bbox, _RISK_HIGHLIGHT_COLOR, f"#{i}", thickness=3)
    return img

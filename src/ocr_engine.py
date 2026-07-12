"""OCR direcionado: lê o texto contido nas zonas de confiança (trust_boundary).

Em vez de rodar OCR na imagem inteira, esta etapa recorta apenas as regiões
detectadas como 'trust_boundary' pelo YOLO e submete só esses recortes ao motor
easyocr. Isso reduz ruído (ícones, setas) e custo de processamento, focando o
OCR nos rótulos das zonas de confiança (ex.: "DMZ", "VPC Pública").
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import easyocr
import numpy as np

from src.vision import _load_image

_reader: easyocr.Reader | None = None

# Altura mínima (px) de um recorte para o OCR ter uma chance razoável de ler o
# texto. Recortes menores são ampliados antes de irem para o easyocr, porque
# rótulos com poucos pixels de altura ficam abaixo do limite prático do motor.
_MIN_CROP_HEIGHT = 200

# Distância normalizada máxima (fração da diagonal do recorte) entre uma leitura
# de OCR e o canto superior-esquerdo para que ela seja aceita como o rótulo da
# zona. Convenção de diagramas AWS/Azure/GCP: o nome da fronteira fica ancorado
# no canto. Leituras mais distantes do canto são componentes internos, não o
# nome da zona — o limiar rejeita rótulos falsos vindos de detecções espúrias.
_LABEL_MAX_CORNER_DIST = 0.15


def _upscale_if_small(crop: np.ndarray) -> np.ndarray:
    height = crop.shape[0]
    if height >= _MIN_CROP_HEIGHT:
        return crop
    scale = _MIN_CROP_HEIGHT / height
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _select_zone_label(readings: list, crop_shape: tuple) -> str:
    """Escolhe o rótulo da zona entre as leituras de OCR por proximidade vetorial.

    Para cada leitura, calcula a distância euclidiana normalizada do seu canto
    superior-esquerdo ao canto (0, 0) do recorte. Retorna o texto da leitura mais
    próxima do canto, desde que dentro de _LABEL_MAX_CORNER_DIST; caso contrário
    (ou sem leituras), retorna "".
    """
    if not readings:
        return ""

    height, width = crop_shape[:2]
    best_dist = None
    best_text = ""

    for bbox, text, _ in readings:
        x_min = min(point[0] for point in bbox)
        y_min = min(point[1] for point in bbox)
        dist = ((x_min / width) ** 2 + (y_min / height) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_text = text

    return best_text if best_dist < _LABEL_MAX_CORNER_DIST else ""


def load_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(
    image: str | Path | bytes | BytesIO,
    trust_boundaries: list[dict],
    to_gray: bool = False,
) -> list[dict]:
    """Recorta cada trust_boundary da imagem e roda OCR apenas nesses recortes.

    Reaproveita cada dict de trust_boundary (com 'class', 'bbox', 'confidence' do
    YOLO) e adiciona as chaves:
      - 'text': todo o texto lido no recorte, concatenado (dado bruto de apoio);
      - 'ocr_confidence': confiança média das leituras do recorte;
      - 'label': nome/legenda oficial da zona, isolado por proximidade ao canto
        superior-esquerdo do recorte (string vazia se nenhuma leitura estiver
        próxima o bastante do canto — ver _select_zone_label).

    O default de to_gray deve casar com o usado em vision.detect() para que os
    recortes venham exatamente da mesma imagem que produziu as bboxes. Ambos
    usam RGB por padrão (to_gray=False); passe to_gray=True em ambos se estiver
    rodando o modelo grayscale legado.
    """
    if not trust_boundaries:
        return []

    reader = load_reader()

    processed = _load_image(image)
    if to_gray:
        processed = processed.convert("L").convert("RGB")
    else:
        processed = processed.convert("RGB")

    img_array = np.array(processed)
    height, width = img_array.shape[:2]

    results: list[dict] = []

    for boundary in trust_boundaries:
        x1, y1, x2, y2 = (int(round(v)) for v in boundary["bbox"])

        # Garante que o recorte fica dentro dos limites da imagem.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        enriched = dict(boundary)

        if x2 <= x1 or y2 <= y1:
            enriched["text"] = ""
            enriched["ocr_confidence"] = 0.0
            enriched["label"] = ""
            results.append(enriched)
            continue

        crop = _upscale_if_small(img_array[y1:y2, x1:x2])
        readings = reader.readtext(crop)

        texts = [text for _, text, _ in readings]
        confidences = [conf for _, _, conf in readings]

        enriched["text"] = " ".join(texts).strip()
        enriched["ocr_confidence"] = (
            round(float(np.mean(confidences)), 4) if confidences else 0.0
        )
        enriched["label"] = _select_zone_label(readings, crop.shape)
        results.append(enriched)

    return results

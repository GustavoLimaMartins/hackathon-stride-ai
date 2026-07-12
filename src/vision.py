"""Triagem de inferência: detecta os elementos de um diagrama de arquitetura via YOLO.

O modelo foi treinado sobre imagens pré-processadas em escala de cinza (grayscale)
pelo Roboflow, então a imagem de entrada passa pela mesma conversão antes da
inferência, mantendo paridade entre o domínio de treino e o de inferência.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

_model: YOLO | None = None


def load_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def _load_image(image: str | Path | bytes | BytesIO) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image)

    data = image.read() if isinstance(image, BytesIO) else image
    return Image.open(BytesIO(data))


def detect(
    image: str | Path | bytes | BytesIO, to_gray: bool = True
) -> tuple[list[dict], list[dict]]:
    """Roda a inferência YOLO e separa as detecções em duas listas.

    Se to_gray=True (padrão), converte a imagem para escala de cinza antes da
    inferência, replicando o pré-processamento aplicado pelo Roboflow no dataset
    de treino e mantendo paridade entre o domínio de treino e o de inferência.

    Retorna (trust_boundaries, other_components), onde a primeira lista contém
    exclusivamente as caixas da classe 'trust_boundary' e a segunda contém todos
    os demais componentes e setas.
    """
    model = load_model()
    processed = _load_image(image)
    if to_gray:
        processed = processed.convert("L").convert("RGB")
    else:
        processed = processed.convert("RGB")

    results = model.predict(np.array(processed), verbose=False)[0]

    trust_boundaries: list[dict] = []
    other_components: list[dict] = []

    for box in results.boxes:
        detection = {
            "class": model.names[int(box.cls)],
            "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
            "confidence": round(float(box.conf), 4),
        }

        if detection["class"] == "trust_boundary":
            trust_boundaries.append(detection)
        else:
            other_components.append(detection)

    return trust_boundaries, other_components

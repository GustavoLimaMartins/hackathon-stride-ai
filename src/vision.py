"""Triagem de inferência: detecta os elementos de um diagrama de arquitetura via YOLO.

Por padrão a imagem de entrada é processada em RGB (cores preservadas). A cor é
um sinal semântico forte em ícones de serviços cloud (AWS/Azure/GCP): compute,
database e storage se distinguem primariamente pela cor. Converter para
grayscale descarta esse sinal e cria um gargalo de informação que tende a
agravar overfitting em datasets pequenos.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Tamanho de inferência: o modelo foi treinado com imgsz=512 (ver train.py).
_INFERENCE_IMGSZ = 512

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
    image: str | Path | bytes | BytesIO, to_gray: bool = False
) -> tuple[list[dict], list[dict]]:
    """Roda a inferência YOLO e separa as detecções em duas listas.

    Por padrão (to_gray=False) a inferência roda sobre a imagem em RGB,
    preservando a cor — sinal semântico forte para ícones de serviços cloud.
    Passe to_gray=True para replicar o pré-processamento grayscale do dataset
    de treino atual.

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

    results = model.predict(
        np.array(processed), imgsz=_INFERENCE_IMGSZ, verbose=False
    )[0]

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

"""Triagem de inferência: detecta os elementos de um diagrama de arquitetura via YOLO.

Por padrão a imagem de entrada é processada em RGB (cores preservadas). A cor é
um sinal semântico forte em ícones de serviços cloud (AWS/Azure/GCP): compute,
database e storage se distinguem primariamente pela cor. Converter para
grayscale descarta esse sinal e cria um gargalo de informação que tende a
agravar overfitting em datasets pequenos.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Tamanho de inferência: o modelo foi treinado com imgsz=1024 (ver train.py).
_INFERENCE_IMGSZ = 1024

# Confiança mínima para uma detecção ser mantida (global, todas as classes).
# 'data_flow' (as setas) é a classe mais ruidosa do modelo: medido no val set,
# mediana de confiança de apenas 0.117 e 77% das detecções abaixo de 0.25 —
# a maior parte são falsos positivos contra o fundo ou confusões com
# 'api_gateway'. Filtrar por confiança ataca o ruído na inferência, sem exigir
# remover a classe do dataset nem retreinar o modelo.
_MIN_CONFIDENCE = 0.25

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


def _resize_meta(
    width: int, height: int, target: int = _INFERENCE_IMGSZ
) -> tuple[float, int, int]:
    """Reproduz o cálculo do letterbox do YOLO para fins de diagnóstico.

    O YOLO não recebe a imagem em 'target'x'target': ele reescala preservando a
    proporção (lado maior -> target) e completa o lado menor com padding cinza.
    Esta função é pura (sem I/O) e apenas expõe o fator de escala e as dimensões
    efetivas pós-escala (antes do padding), para registro/observabilidade. Não
    redimensiona nada — a inferência continua entregando a imagem original ao
    predict(..., imgsz=target).

    Retorna (scale, scaled_width, scaled_height), onde scale > 1 indica upscale
    (imagem menor que target, risco de borrão) e scale < 1 indica downscale.
    """
    scale = target / max(width, height)
    return scale, round(width * scale), round(height * scale)


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
    os demais componentes e setas. Detecções com confiança abaixo de
    _MIN_CONFIDENCE (25%, global para todas as classes) são descartadas — filtro
    aplicado nativamente pelo YOLO via o parâmetro 'conf' do predict().
    """
    model = load_model()
    processed = _load_image(image)
    if to_gray:
        processed = processed.convert("L").convert("RGB")
    else:
        processed = processed.convert("RGB")

    # Diagnóstico: expõe o resize que o YOLO fará internamente (letterbox para
    # _INFERENCE_IMGSZ). Não altera a inferência — só registra escala e sentido.
    if logger.isEnabledFor(logging.DEBUG):
        width, height = processed.size
        scale, scaled_w, scaled_h = _resize_meta(width, height)
        direction = "upscale" if scale > 1 else "downscale" if scale < 1 else "sem escala"
        logger.debug(
            "YOLO resize (%s): original %dx%d -> %dx%d (escala %.3f) "
            "-> letterbox %dx%d",
            direction,
            width,
            height,
            scaled_w,
            scaled_h,
            scale,
            _INFERENCE_IMGSZ,
            _INFERENCE_IMGSZ,
        )

    results = model.predict(
        np.array(processed),
        imgsz=_INFERENCE_IMGSZ,
        conf=_MIN_CONFIDENCE,
        verbose=False,
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

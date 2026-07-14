"""Script de treinamento do modelo de visão computacional (YOLOv8 Nano).

Fase 2 do projeto: treina um modelo supervisionado capaz de detectar os
elementos dos diagramas de arquitetura (7 classes definidas em dataset/data.yaml).

Uso:
    .venv\\Scripts\\python.exe train.py

Ao final, os pesos treinados ficam em runs/detect/train/weights/best.pt.
Copie best.pt para models/ para versionar o modelo final do projeto.
"""

from pathlib import Path

from ultralytics import YOLO

# Raiz do projeto (diretório onde este script está localizado).
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"


def main() -> None:
    # Modelo base pré-treinado YOLOv8 Nano (baixado automaticamente na 1ª execução).
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1024,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="train",
    )


if __name__ == "__main__":
    main()

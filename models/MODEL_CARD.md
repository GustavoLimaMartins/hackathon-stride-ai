# Model Card — models/best.pt

## Identificação

| Campo | Valor |
|---|---|
| Arquitetura | YOLOv8n (nano) |
| Framework | Ultralytics YOLO v8.4.92 |
| Parâmetros treináveis | 3.012.213 (~3,0 M) |
| Tamanho do arquivo | 6,1 MB |
| Tarefa | Detecção de objetos (bounding boxes) |
| Classes (nc=7) | `actor`, `api_gateway`, `compute`, `data_flow`, `database_storage`, `network_security`, `trust_boundary` |
| Data de treino | 2026-07-14 |
| Commit do repositório na geração | `38beb41` — "feat(vision): adicionar filtro de confiança mínima (25%) na inferência YOLO" (branch `main`) |
| Run de origem | `runs/detect/train-7` |

## Procedência dos dados

| Campo | Valor |
|---|---|
| Fonte do dataset | Roboflow — projeto `hackathon-vp4zy` (workspace `gustavo-lima-vprcy`), versão 12 |
| Licença do dataset | CC BY 4.0 |
| Imagens de treino | 420 arquivos (44 imagens-base únicas antes de augmentation) |
| Imagens de validação | 13 imagens, 614 instâncias anotadas |
| Imagens de teste | 6 imagens |

## Configuração de treinamento

| Hiperparâmetro | Valor |
|---|---|
| Modelo base | `yolov8n.pt` (pré-treinado, fine-tuning) |
| Épocas | 100 (patience=100, sem early stopping efetivo) |
| Resolução de entrada (imgsz) | 1024×1024 |
| Batch size | 16 |
| Otimizador | auto (seleção automática do Ultralytics) |
| Learning rate inicial (lr0) | 0,01 |
| Learning rate final (lrf) | 0,01 |
| Momentum | 0,937 |
| Weight decay | 0,0005 |
| Warmup (épocas) | 3 |
| Seed | 0 (determinístico) |
| AMP (precisão mista) | ativado |
| Pesos de loss | box=7.5, cls=0.5, dfl=1.5 |

### Augmentation aplicada
HSV (h=0.015, s=0.7, v=0.4); translação=0.1; escala=0.5; flip horizontal=0.5 (flip vertical desativado); mosaic=1.0 (desativado nas últimas 10 épocas via `close_mosaic`); mixup e copy-paste desativados.

## Métricas de validação (época selecionada: 40 de 100)

Avaliadas em `dataset/valid` (imgsz=1024), 13 imagens / 614 instâncias:

| Classe | Precisão | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **todas** | **0,603** | **0,447** | **0,496** | **0,239** |
| api_gateway | 0,814 | 0,611 | 0,742 | 0,280 |
| trust_boundary | 0,627 | 0,759 | 0,727 | 0,522 |
| database_storage | 0,872 | 0,485 | 0,624 | 0,257 |
| compute | 0,628 | 0,532 | 0,541 | 0,235 |
| actor | 0,510 | 0,265 | 0,404 | 0,152 |
| network_security | 0,503 | 0,381 | 0,376 | 0,210 |
| data_flow | 0,266 | 0,078 | 0,061 | 0,016 |

*Valores extraídos de `train_metrics` embutido no checkpoint (`best.pt`) e reconfirmados via validação explícita (`YOLO('models/best.pt').val(...)`); consistentes com `runs/detect/train-7/results.csv`, época 40, e com a matriz de confusão (`runs/detect/train-7/confusion_matrix.png`).*

## Limitações conhecidas

- **`data_flow` (setas/conexões) tem recall muito baixo (0,078)**: análise da matriz de confusão mostra que ~85% das instâncias verdadeiras dessa classe não são detectadas (não há confusão sistemática com outra classe). Causa provável: geometria de objeto fino/alongado, combinada com volume insuficiente de exemplos de treino (44 imagens-base).
- **Conjunto de validação reduzido (13 imagens)**: as métricas por classe têm variância alta e devem ser lidas como indicação qualitativa de desempenho relativo entre classes, não como valores de precisão estatística fina.
- **Instabilidade entre execuções de treino**: repetições com hiperparâmetros idênticos, variando apenas dataset/imgsz, produziram mAP50 entre ~0,27 e ~0,50 — atribuído ao tamanho pequeno do dataset-base.
- Inferência em produção usa `imgsz=1024` explícito (deve corresponder ao imgsz de treino) e um filtro de confiança mínima de 25% aplicado na etapa de detecção.

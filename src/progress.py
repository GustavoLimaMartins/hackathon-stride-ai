"""Estimativa de progresso do processamento do relatório STRIDE.

O pipeline tem 5 etapas sequenciais de durações muito diferentes — a análise do
LLM sozinha leva minutos, enquanto as demais levam segundos. Uma barra de
progresso ingênua congelaria na etapa do LLM (chamada bloqueante). Para dar
feedback contínuo, este módulo infere o avanço a partir do TEMPO MÉDIO histórico
de cada etapa: cada etapa ocupa uma fatia da barra proporcional ao seu tempo
esperado, e dentro de uma etapa em andamento o preenchimento interpola pelo tempo
decorrido / tempo esperado.

Os tempos são persistidos em .stride_timings.json (média móvel entre execuções),
calibrando a estimativa ao hardware do usuário. Na primeira execução usam-se os
DEFAULT_TIMINGS embutidos.

Este módulo é puro (não importa Streamlit) para ser testável isoladamente.
"""

from __future__ import annotations

import json
from pathlib import Path

# Ordem canônica das etapas (chave interna -> rótulo legível exibido ao usuário).
STEPS: list[tuple[str, str]] = [
    ("detect", "Detectando componentes (YOLO)"),
    ("ocr_zones", "Lendo rótulos das zonas (OCR)"),
    ("ocr_components", "Associando rótulos aos componentes (OCR)"),
    ("graph", "Montando o grafo hierárquico"),
    ("llm", "Analisando ameaças (STRIDE)"),
]

STEP_KEYS: list[str] = [key for key, _ in STEPS]
STEP_LABELS: dict[str, str] = dict(STEPS)

# Estimativas iniciais (segundos) para o cold start — substituídas pelos tempos
# reais a partir da 1ª execução. O LLM domina o total; os demais são rápidos.
DEFAULT_TIMINGS: dict[str, float] = {
    "detect": 15.0,
    "ocr_zones": 10.0,
    "ocr_components": 10.0,
    "graph": 1.0,
    "llm": 90.0,
}

# Peso do novo valor na média móvel exponencial (0<alpha<=1). 0.5 = o ETA
# acompanha de perto as execuções recentes (metade do valor vem da última
# medição), evitando que o tempo estimado fique defasado quando a duração do LLM
# tem tendência de alta — sem deixar um único outlier dominar a estimativa.
_EMA_ALPHA = 0.5

# Arquivo de calibração na raiz do projeto (git-ignored).
_TIMINGS_PATH = Path(__file__).resolve().parent.parent / ".stride_timings.json"


def _timings_path() -> Path:
    return _TIMINGS_PATH


def load_timings() -> dict[str, float]:
    """Tempos médios por etapa; defaults quando o arquivo falta ou está corrompido.

    Sempre retorna um dict com TODAS as chaves de STEP_KEYS (mescla o que foi lido
    sobre os defaults), para o estimador nunca receber uma etapa sem tempo.
    """
    timings = dict(DEFAULT_TIMINGS)
    path = _timings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in STEP_KEYS:
            value = data.get(key)
            if isinstance(value, (int, float)) and value > 0:
                timings[key] = float(value)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        pass  # cai nos defaults — arquivo ausente/ilegível não deve quebrar o app
    return timings


def save_timing(step: str, seconds: float) -> None:
    """Atualiza a média móvel exponencial da duração de 'step' e persiste.

    Ignora entradas inválidas (etapa desconhecida ou tempo não-positivo) e falhas
    de escrita — a calibração é um "nice to have" que jamais deve interromper o
    fluxo do usuário.
    """
    if step not in STEP_KEYS or not seconds or seconds <= 0:
        return
    timings = load_timings()
    old = timings.get(step, DEFAULT_TIMINGS.get(step, seconds))
    timings[step] = (1 - _EMA_ALPHA) * old + _EMA_ALPHA * float(seconds)
    try:
        _timings_path().write_text(
            json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


class StepEstimator:
    """Converte (etapa, tempo decorrido na etapa) numa fração global da barra [0,1].

    Cada etapa ocupa uma fatia da barra proporcional ao seu tempo médio sobre a
    soma de todos os tempos. Dentro de uma etapa em andamento, o preenchimento
    interpola por elapsed/eta, limitado a 0.99 da fatia (só chega ao fim quando a
    etapa é marcada como concluída), de modo que a barra nunca "estoura" a fatia
    antes da hora nem regride.
    """

    def __init__(self, timings: dict[str, float] | None = None) -> None:
        self._timings = {**DEFAULT_TIMINGS, **(timings or {})}
        total = sum(self._timings[k] for k in STEP_KEYS) or 1.0
        # Fração acumulada [start, end) de cada etapa na barra.
        self._bounds: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for key in STEP_KEYS:
            width = self._timings[key] / total
            self._bounds[key] = (acc, acc + width)
            acc += width

    def step_start(self, step: str) -> float:
        """Fração da barra no início da etapa."""
        return self._bounds[step][0]

    def step_end(self, step: str) -> float:
        """Fração da barra no fim da etapa (início da próxima)."""
        return self._bounds[step][1]

    def step_progress(self, step: str, elapsed_in_step: float) -> float:
        """Progresso LOCAL da etapa em [0,0.99]: elapsed / tempo médio esperado.

        É o quanto do tempo médio daquela etapa já decorreu — limitado a 0.99 até
        a etapa concluir, para o texto não mostrar 100% antes da hora.
        """
        eta = self._timings[step]
        if eta <= 0:
            return 0.0
        return min(max(elapsed_in_step / eta, 0.0), 0.99)

    def step_percent(self, step: str, elapsed_in_step: float) -> int:
        """Progresso local da etapa como inteiro 0..99 (para exibir 'x%')."""
        return int(self.step_progress(step, elapsed_in_step) * 100)

    def fraction(self, step: str, elapsed_in_step: float) -> float:
        """Fração global estimada com a etapa 'step' rodando há 'elapsed_in_step's."""
        start, end = self._bounds[step]
        # Interpola dentro da fatia, sem alcançar o fim antes de concluir.
        return start + (end - start) * self.step_progress(step, elapsed_in_step)


def format_elapsed(seconds: float) -> str:
    """Segundos -> 'm:ss' para o texto do progresso."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"

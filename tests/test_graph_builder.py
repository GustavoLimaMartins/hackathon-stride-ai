"""Testes da lógica de grafo — foco no sinal de proximidade (proximity_hints).

A proximidade intra-zona é um enriquecimento puramente geométrico: dá para
validá-la com fixtures manuais de dicts, sem depender do YOLO/OCR. Os dicts aqui
reproduzem o formato que 'to_json' consome (class, bbox, confidence, name).

Convenção de bbox: [x1, y1, x2, y2] em pixels absolutos, como o YOLO entrega.
Nas fixtures usamos uma imagem 1000x1000 (diagonal ~1414px); com
_PROXIMITY_MAX_DIST_FRAC=0.12 o teto de distância é ~170px. "Perto" = dezenas de
px entre centróides; "longe" = colocar o componente num canto oposto.
"""

from __future__ import annotations

from src.graph_builder import proximity_hints, to_json

IMG = (1000, 1000)


def _comp(cls: str, bbox: list, name: str = "", conf: float = 0.8) -> dict:
    return {"class": cls, "bbox": bbox, "confidence": conf, "name": name}


def _boundary(bbox: list, label: str = "", conf: float = 0.9) -> dict:
    return {"class": "trust_boundary", "bbox": bbox, "confidence": conf, "label": label}


def _pairs(graph: dict) -> set:
    """Conjunto de pares (frozenset de ids) presentes em proximity_hints."""
    return {frozenset((h["source"], h["target"])) for h in graph["proximity_hints"]}


# 1. Dois componentes próximos na mesma zona -> gera 1 hint.
def test_two_close_components_same_zone_produce_one_hint():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [
        _comp("compute", [100, 100, 140, 140], "Lambda"),
        _comp("database_storage", [150, 150, 190, 190], "RDS"),
    ]
    graph = to_json(tb, comps, image_size=IMG)
    assert _pairs(graph) == {frozenset(("c0", "c1"))}
    assert graph["proximity_hints"][0]["distance_frac"] > 0


# 2. Dois componentes distantes (acima do teto) na mesma zona -> nenhum hint.
def test_distant_components_same_zone_produce_no_hint():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [
        _comp("compute", [10, 10, 50, 50]),
        _comp("database_storage", [950, 950, 990, 990]),  # canto oposto
    ]
    graph = to_json(tb, comps, image_size=IMG)
    assert graph["proximity_hints"] == []


# 3. Componentes próximos em zonas diferentes -> nenhum hint (intra-zona).
def test_close_components_different_zones_produce_no_hint():
    tb = [
        _boundary([0, 0, 200, 1000], "A"),
        _boundary([200, 0, 400, 1000], "B"),
    ]
    comps = [
        _comp("compute", [150, 100, 190, 140]),  # zona A
        _comp("compute", [210, 100, 250, 140]),  # zona B (perto, mas outra zona)
    ]
    graph = to_json(tb, comps, image_size=(400, 1000))
    assert graph["proximity_hints"] == []


# 4. Par que já tem data_flow -> hint suprimido (dedup complementar).
def test_pair_with_data_flow_is_deduped():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [
        _comp("compute", [100, 100, 140, 140], "A"),
        _comp("compute", [150, 150, 190, 190], "B"),
        # seta cujos cantos caem sobre A e B -> connect_data_flows liga c0-c1
        _comp("data_flow", [120, 120, 170, 170]),
    ]
    graph = to_json(tb, comps, image_size=IMG)
    assert {frozenset((f["source"], f["target"])) for f in graph["data_flows"]} == {
        frozenset(("c0", "c1"))
    }
    assert graph["proximity_hints"] == []  # já coberto pela seta


# 5. Simetria: a proximidade a<->b produz um único hint (não a->b e b->a).
def test_symmetric_pair_produces_single_hint():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [
        _comp("compute", [100, 100, 140, 140]),
        _comp("compute", [150, 150, 190, 190]),
    ]
    graph = to_json(tb, comps, image_size=IMG)
    assert len(graph["proximity_hints"]) == 1


# 6. Componentes sem zona (unassigned) -> nenhum hint.
def test_unassigned_components_produce_no_hint():
    # Sem boundary nenhuma: os dois componentes ficam unassigned.
    comps = [
        _comp("compute", [100, 100, 140, 140]),
        _comp("database_storage", [150, 150, 190, 190]),
    ]
    graph = to_json([], comps, image_size=IMG)
    assert graph["unassigned_components"]  # de fato caíram em unassigned
    assert graph["proximity_hints"] == []


# 7. to_json sem image_size usa o fallback do envelope das bboxes e ainda funciona.
def test_to_json_without_image_size_uses_bbox_envelope():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [
        _comp("compute", [100, 100, 140, 140]),
        _comp("database_storage", [150, 150, 190, 190]),
    ]
    graph = to_json(tb, comps)  # sem image_size
    assert "proximity_hints" in graph
    # o envelope aqui é a própria boundary (0..1000), diagonal ~1414 -> o par
    # próximo continua abaixo do teto e gera o hint.
    assert _pairs(graph) == {frozenset(("c0", "c1"))}


# 8. Centróides quase sobrepostos (detecção duplicada) -> descartados pelo piso.
def test_near_overlapping_centroids_are_discarded():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    # Duas bboxes praticamente na mesma posição: centróides a ~2px um do outro,
    # bem abaixo do piso (_PROXIMITY_MIN_DIST_FRAC * diagonal ~= 14px em 1000x1000).
    comps = [
        _comp("compute", [100, 100, 140, 140]),
        _comp("compute", [102, 102, 142, 142]),
    ]
    graph = to_json(tb, comps, image_size=IMG)
    assert graph["proximity_hints"] == []


# Extra: setas nunca são nós de proximidade, e o KNN respeita _PROXIMITY_K.
def test_data_flow_class_is_never_a_proximity_node():
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [
        _comp("compute", [100, 100, 140, 140]),
        _comp("data_flow", [145, 145, 175, 175]),  # perto, mas é seta
    ]
    graph = to_json(tb, comps, image_size=IMG)
    assert graph["proximity_hints"] == []  # só há 1 nó conectável na zona


def test_proximity_hints_respects_k_neighbors():
    # 5 componentes bem próximos em fila; com _PROXIMITY_K=2 cada nó liga a no
    # máximo 2 vizinhos, então nenhum nó aparece em mais de ~2 hints.
    tb = [_boundary([0, 0, 1000, 1000], "VPC")]
    comps = [_comp("compute", [100 + i * 30, 100, 130 + i * 30, 130]) for i in range(5)]
    graph = to_json(tb, comps, image_size=IMG)
    counts: dict = {}
    for h in graph["proximity_hints"]:
        counts[h["source"]] = counts.get(h["source"], 0) + 1
        counts[h["target"]] = counts.get(h["target"], 0) + 1
    # nenhum componente vira nó de proximidade além do que K permite por ambos os
    # lados (K de saída + ser escolhido por vizinhos) — sanidade: grafo não explode.
    assert graph["proximity_hints"]  # produziu algo
    assert max(counts.values()) <= 4

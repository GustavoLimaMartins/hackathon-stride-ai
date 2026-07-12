"""Engenharia espacial: associa componentes às zonas de confiança que os contêm.

A matemática de contenção geométrica pega o centróide de cada componente
(servidores, bancos de dados, gateways, setas) e verifica em qual trust_boundary
esse ponto cai. Isso constrói a hierarquia "componente pertence a zona", base
para o mapeamento STRIDE: um componente na zona errada (ou em nenhuma) é sinal
de risco de segurança.
"""

from __future__ import annotations

from collections import defaultdict

# Componentes que funcionam como "nós" conectáveis por setas. 'data_flow' (a
# própria seta) e 'trust_boundary' (a zona) não são nós de conexão.
_CONNECTABLE_CLASSES = {
    "actor",
    "api_gateway",
    "compute",
    "database_storage",
    "network_security",
}


def _centroid(bbox: list) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _display_id(boundary: dict, index: int) -> str:
    """Identificador da zona: o label lido pelo OCR, ou um fallback posicional."""
    label = boundary.get("label", "")
    return label if label else f"boundary_{index}"


def assign_trust_boundaries(
    components: list[dict], trust_boundaries: list[dict]
) -> list[dict]:
    """Associa cada componente à trust_boundary mais específica que o contém.

    Para cada componente, calcula o centróide da sua bbox e verifica quais
    trust_boundaries o contêm (teste ponto-em-retângulo). Havendo mais de uma
    (zonas aninhadas), escolhe a de menor área — a zona mais específica.

    Retorna novos dicts (sem mutar os originais) com o campo 'trust_boundary'
    adicionado: o label/identificador da zona associada, ou None se o centróide
    não cai em nenhuma boundary.
    """
    # Pré-calcula área e identificador de exibição de cada boundary uma só vez.
    boundary_info = []
    for index, boundary in enumerate(trust_boundaries):
        x1, y1, x2, y2 = boundary["bbox"]
        area = (x2 - x1) * (y2 - y1)
        boundary_info.append((x1, y1, x2, y2, area, _display_id(boundary, index)))

    results: list[dict] = []

    for component in components:
        cx, cy = _centroid(component["bbox"])

        # Coleta as boundaries que contêm o centróide e fica com a de menor área.
        best_area = None
        best_id = None
        for bx1, by1, bx2, by2, area, display_id in boundary_info:
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if best_area is None or area < best_area:
                    best_area = area
                    best_id = display_id

        enriched = dict(component)
        enriched["trust_boundary"] = best_id
        results.append(enriched)

    return results


def group_by_boundary(components: list[dict]) -> dict:
    """Agrupa componentes pela zona de confiança a que pertencem (herança semântica).

    View agregada derivada de assign_trust_boundaries(): cada componente já traz o
    campo 'trust_boundary', e aqui os "filhos" são coletados por zona. Componentes
    sem zona ficam sob a chave None.

    Retorna dict {label_da_zona (ou None): [componentes...]}.
    """
    grouped: dict = defaultdict(list)
    for component in components:
        grouped[component.get("trust_boundary")].append(component)
    return dict(grouped)


def connect_data_flows(components: list[dict]) -> list[dict]:
    """Liga cada seta (data_flow) aos dois nós mais próximos das suas extremidades.

    Para cada componente da classe 'data_flow', calcula os 4 cantos da sua bbox e
    encontra os dois nós conectáveis cujos centróides estão mais próximos de
    qualquer um desses cantos. A conexão é não-direcionada: o YOLO não distingue
    origem de destino, então retornar um sentido seria adivinhar.

    Os nós são referenciados pelo índice posicional na lista 'components' recebida.
    Retorna lista de {'components': [idx_a, idx_b], 'confidence': float}, uma por
    seta que tenha ao menos 2 nós candidatos.
    """
    node_indices = [
        i for i, c in enumerate(components) if c["class"] in _CONNECTABLE_CLASSES
    ]
    node_centroids = {i: _centroid(components[i]["bbox"]) for i in node_indices}

    connections: list[dict] = []

    for arrow in components:
        if arrow["class"] != "data_flow":
            continue
        if len(node_indices) < 2:
            continue

        x1, y1, x2, y2 = arrow["bbox"]
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

        # Distância de cada nó = menor distância entre seu centróide e um dos cantos.
        distances = []
        for idx in node_indices:
            cx, cy = node_centroids[idx]
            min_dist = min(
                ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 for ax, ay in corners
            )
            distances.append((min_dist, idx))

        distances.sort(key=lambda d: d[0])
        idx_a, idx_b = distances[0][1], distances[1][1]

        connections.append(
            {"components": [idx_a, idx_b], "confidence": arrow["confidence"]}
        )

    return connections


def _clean_component(component: dict, comp_id: str) -> dict:
    """Objeto de componente para o JSON: id + campos essenciais, sem redundância.

    Omite 'trust_boundary' (a posição do componente dentro do JSON já expressa a
    zona) e 'data_flow' não chega aqui — setas viram conexões, não nós.
    """
    return {
        "id": comp_id,
        "class": component["class"],
        "bbox": [float(v) for v in component["bbox"]],
        "confidence": float(component["confidence"]),
    }


def to_json(trust_boundaries: list[dict], components: list[dict]) -> dict:
    """Compila o grafo completo num objeto JSON hierárquico serializável.

    Orquestra as etapas já implementadas (contenção, agrupamento, conexões) e
    monta a estrutura final: zonas de confiança nomeadas, cada uma com seus
    componentes aninhados, mais os fluxos de dados que ligam componentes por id.

    Estrutura:
      {
        "trust_boundaries": [{"id", "label", "bbox", "components": [...]}],
        "unassigned_components": [...],
        "data_flows": [{"source", "target", "confidence"}]
      }

    Cada componente recebe um id estável ("c0", "c1", ...) baseado na sua posição
    na lista de entrada, usado tanto no aninhamento quanto nas conexões. As
    conexões são NÃO-DIRECIONADAS: 'source'/'target' são apenas as duas pontas
    (o YOLO não distingue origem de destino), não um sentido de fluxo inferido.
    """
    assigned = assign_trust_boundaries(components, trust_boundaries)

    # Id estável por componente, pela posição na lista (mesma base dos índices
    # que connect_data_flows() devolve).
    comp_ids = [f"c{i}" for i in range(len(assigned))]

    # Conexões: traduz os índices posicionais para os ids de componente.
    connections = connect_data_flows(assigned)
    data_flows = [
        {
            "source": comp_ids[conn["components"][0]],
            "target": comp_ids[conn["components"][1]],
            "confidence": float(conn["confidence"]),
        }
        for conn in connections
    ]

    # Mapa: display_id da zona -> objeto boundary do JSON (para aninhar filhos).
    boundary_objs: dict = {}
    boundaries_json: list[dict] = []
    for index, boundary in enumerate(trust_boundaries):
        display_id = _display_id(boundary, index)
        obj = {
            "id": f"b{index}",
            "label": boundary.get("label", ""),
            "bbox": [float(v) for v in boundary["bbox"]],
            "components": [],
        }
        boundary_objs[display_id] = obj
        boundaries_json.append(obj)

    # Aninha cada componente (exceto setas) na sua zona, ou em unassigned.
    unassigned: list[dict] = []
    for i, component in enumerate(assigned):
        if component["class"] == "data_flow":
            continue
        clean = _clean_component(component, comp_ids[i])
        zone = component["trust_boundary"]
        if zone is not None and zone in boundary_objs:
            boundary_objs[zone]["components"].append(clean)
        else:
            unassigned.append(clean)

    return {
        "trust_boundaries": boundaries_json,
        "unassigned_components": unassigned,
        "data_flows": data_flows,
    }

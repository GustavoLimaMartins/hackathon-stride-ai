"""Schema estruturado do parecer STRIDE (saída do LLM 'analyst').

Em vez de Markdown livre, o analyst devolve um StrideReport: uma lista de riscos
tipados, cada um ancorado num elemento do grafo (componente, fluxo ou zona) pelo
seu id. Esse vínculo id -> bounding box é o que permite gerar, para cada risco, o
recorte visual do ponto exato do diagrama que precisa de intervenção
(rastreabilidade visual). O schema é injetado no LLM via
ChatOpenAI.with_structured_output (ver stride_engine.load_analyst_structured).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Ordem de severidade (crítica primeiro) — usada para ordenar os riscos na UI.
SEVERITY_ORDER: dict[str, int] = {
    "Crítica": 0,
    "Alta": 1,
    "Média": 2,
    "Baixa": 3,
}


class Risk(BaseModel):
    """Um risco STRIDE ancorado a um elemento do grafo de arquitetura."""

    target_type: Literal["component", "flow", "boundary"] = Field(
        description=(
            "Tipo do elemento afetado no diagrama: 'component' (um objeto: actor, "
            "compute, database_storage, api_gateway, network_security), 'flow' (um "
            "fluxo de dados entre dois componentes) ou 'boundary' (uma zona/região "
            "de confiança inteira)."
        )
    )
    target_id: str = Field(
        description=(
            "Id do elemento afetado no grafo. Para 'component' use o id do componente "
            "(ex.: 'c4'). Para 'boundary' use o id da zona (ex.: 'b1'). Para 'flow' "
            "use o id de UMA das pontas (ex.: 'c2') e preencha também flow_source_id "
            "e flow_target_id."
        )
    )
    flow_source_id: str | None = Field(
        default=None,
        description=(
            "Somente quando target_type == 'flow': id do componente de uma ponta do "
            "fluxo (ex.: 'c2'). Deixe vazio para component/boundary."
        ),
    )
    flow_target_id: str | None = Field(
        default=None,
        description=(
            "Somente quando target_type == 'flow': id do componente da outra ponta "
            "do fluxo (ex.: 'c5'). Deixe vazio para component/boundary."
        ),
    )
    stride_category: str = Field(
        description=(
            "Categoria STRIDE: Spoofing, Tampering, Repudiation, Information "
            "Disclosure, Denial of Service ou Elevation of Privilege."
        )
    )
    elemento_afetado: str = Field(
        description=(
            "Nome legível do elemento afetado — o 'name' real lido do diagrama "
            "quando houver (ex.: 'Amazon Lambda'), senão '<classe> (<id>)'."
        )
    )
    justificativa: str = Field(
        description="Motivo técnico da classificação, ancorado na estrutura do grafo."
    )
    impacto: str = Field(
        description="Consequência potencial concreta caso a ameaça se realize."
    )
    severidade: Literal["Baixa", "Média", "Alta", "Crítica"] = Field(
        description="Severidade do risco."
    )
    contramedida: str = Field(
        description="Mitigação técnica específica e prescritiva (nunca genérica)."
    )


class StrideReport(BaseModel):
    """Parecer STRIDE completo: a lista de riscos identificados no diagrama."""

    risks: list[Risk] = Field(
        default_factory=list,
        description="Todos os riscos STRIDE identificados, um por ameaça+elemento.",
    )


def severity_rank(severidade: str) -> int:
    """Índice de ordenação de uma severidade (crítica=0). Desconhecida vai por último."""
    return SEVERITY_ORDER.get(severidade, len(SEVERITY_ORDER))


class RiskGroup(BaseModel):
    """Riscos consolidados que afetam o MESMO elemento arquitetural.

    A unidade de exibição do relatório é o elemento (um componente, um fluxo ou
    uma zona), não a ameaça individual: um API Gateway com quatro categorias
    STRIDE vira um único grupo — uma imagem, uma tabela com quatro linhas. Isso
    elimina a repetição de recortes idênticos e encurta o relatório.
    """

    target_type: Literal["component", "flow", "boundary"]
    # Um risco representativo do grupo (o de maior severidade), usado para
    # localizar/destacar o elemento no diagrama e nomear o bloco.
    representative: Risk
    # Todos os riscos do grupo, já ordenados por severidade (crítica primeiro).
    risks: list[Risk]

    @property
    def worst_severity_rank(self) -> int:
        """Rank da pior severidade do grupo — base para ordenar os blocos."""
        return min((severity_rank(r.severidade) for r in self.risks), default=len(SEVERITY_ORDER))


def _risk_group_key(risk: Risk) -> tuple:
    """Chave que identifica 'o mesmo elemento/recorte' de um risco.

    - component / boundary: o próprio target_id.
    - flow: o par NÃO-ORDENADO das pontas — a mesma comunicação a<->b e b<->a
      cai no mesmo grupo, independentemente da ordem devolvida pelo LLM.
    """
    if risk.target_type == "flow":
        pair = frozenset(
            p for p in (risk.flow_source_id, risk.flow_target_id) if p
        ) or frozenset({risk.target_id})
        return ("flow", pair)
    return (risk.target_type, risk.target_id)


def group_risks(risks: list[Risk]) -> list[RiskGroup]:
    """Consolida riscos por elemento arquitetural e ordena os grupos por gravidade.

    Agrupa pela chave de _risk_group_key (mesmo componente / fluxo / zona). Dentro
    de cada grupo, ordena os riscos por severidade (crítica primeiro). Os grupos
    saem ordenados pela pior severidade que contêm — o elemento com o risco mais
    crítico aparece primeiro no relatório. Preserva a ordem de descoberta entre
    grupos de mesma severidade (estabilidade).
    """
    order: list[tuple] = []
    buckets: dict[tuple, list[Risk]] = {}
    for risk in risks:
        key = _risk_group_key(risk)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(risk)

    groups: list[RiskGroup] = []
    for key in order:
        ordered = sorted(buckets[key], key=lambda r: severity_rank(r.severidade))
        groups.append(
            RiskGroup(
                target_type=ordered[0].target_type,
                representative=ordered[0],  # maior severidade do grupo
                risks=ordered,
            )
        )

    groups.sort(key=lambda g: g.worst_severity_rank)
    return groups

"""Frontend Streamlit: a vitrine da análise STRIDE de diagramas de arquitetura.

Esta é a camada de apresentação do projeto. O backend (Fases 2-4) já entrega o
pipeline completo: visão computacional (YOLO) detecta os componentes do diagrama,
OCR lê os rótulos das zonas de confiança, a engenharia espacial monta um grafo
hierárquico em JSON e o LLM ('analyst') produz o parecer STRIDE com contramedidas.

Microtarefa 5.2: orquestração síncrona — ao clicar em "Analisar", a imagem
enviada percorre o pipeline (detect → extract_text → to_json → analyst) e o
parecer STRIDE é renderizado em Markdown. A UI apenas encadeia funções já
implementadas e validadas nas fases anteriores; não há lógica de domínio nova.

Execução:
    streamlit run src/main.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# 'streamlit run src/main.py' coloca a pasta src/ no sys.path (não a raiz), então
# os imports 'from src.xxx' abaixo não seriam resolvidos. Garantir a raiz do
# projeto no path faz o app rodar de qualquer diretório de trabalho.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Carrega o .env da raiz do projeto para o ambiente, tornando a OPENAI_API_KEY
# visível ao stride_engine. Sem isso, o app roda mas a análise STRIDE falha com
# "OPENAI_API_KEY não configurada" (o load_llm levanta RuntimeError).
load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph_builder import to_json
from src.ocr_engine import assign_component_names, extract_text
from src.prompts import STRIDE_ANALYST_SYSTEM_PROMPT, build_stride_user_message
from src.stride_engine import load_analyst_structured
from src.stride_models import group_risks
from src.vision import _load_image, detect
from src.visual_report import (
    draw_numbered_overlay,
    draw_overlay,
    highlight_element,
    legend_items,
    resolve_graph_index,
)

# set_page_config deve ser a primeira chamada Streamlit do script.
st.set_page_config(
    page_title="STRIDE-AI — Análise de Ameaças em Diagramas de Arquitetura Cloud",
    page_icon="🛡️",
    layout="wide",
)

def _proximity_lines(graph: dict) -> list:
    """Traduz os proximity_hints do grafo em pares de centróides para o overlay.

    Os hints referenciam componentes por id ("c0", "c1"...); aqui mapeamos cada
    id para o centróide da sua bbox (varrendo componentes aninhados nas zonas e
    os unassigned) e devolvemos os pares de pontos que draw_overlay liga com a
    linha tracejada. Assim visual_report não precisa conhecer o esquema de ids.
    """
    centroids: dict = {}
    boundaries = graph.get("trust_boundaries", [])
    nested = (c for b in boundaries for c in b.get("components", []))
    for comp in (*nested, *graph.get("unassigned_components", [])):
        x1, y1, x2, y2 = comp["bbox"]
        centroids[comp["id"]] = ((x1 + x2) / 2, (y1 + y2) / 2)

    lines = []
    for hint in graph.get("proximity_hints", []):
        src, tgt = centroids.get(hint["source"]), centroids.get(hint["target"])
        if src and tgt:
            lines.append((src, tgt))
    return lines


def _element_location(group, graph: dict) -> str:
    """Zona de confiança onde o elemento do grupo está — o 'onde intervir'.

    Para uma zona (boundary), é o próprio nome/label dela. Para um componente ou
    fluxo, é a trust_boundary que contém o(s) componente(s): procura em qual
    boundary do grafo o id aparece aninhado. Retorna "Não associado a zona" quando
    o componente é unassigned, ou "-" quando não há como determinar.
    """
    rep = group.representative
    boundaries = graph.get("trust_boundaries", [])

    if group.target_type == "boundary":
        zone = resolve_graph_index(graph).get(rep.target_id)
        return (zone.get("label") if zone else "") or rep.target_id

    # component/flow: acha a zona que contém o id de referência.
    ref_ids = [
        i for i in (rep.target_id, rep.flow_source_id, rep.flow_target_id) if i
    ]
    for boundary in boundaries:
        member_ids = {c["id"] for c in boundary.get("components", [])}
        if any(rid in member_ids for rid in ref_ids):
            return boundary.get("label") or boundary["id"]

    # não está em nenhuma zona: só é "unassigned" se o id existe no grafo.
    index = resolve_graph_index(graph)
    if any(rid in index for rid in ref_ids):
        return "Não associado a zona"
    return "-"


st.title("🛡️ STRIDE-AI — Análise de Ameaças em Diagramas de Arquitetura Cloud")

st.markdown(
    """
    Envie um **diagrama de arquitetura cloud** (AWS, Azure ou GCP) e receba uma
    análise de segurança automatizada pela metodologia **STRIDE**.

    Nos bastidores, a ferramenta combina três etapas: **visão computacional**
    (YOLO) para detectar os componentes do diagrama, **OCR** para ler os rótulos
    das zonas de confiança e um **LLM especialista** que atua como arquiteto
    DevSecOps — mapeando ameaças (Spoofing, Tampering, Repudiation, Information
    Disclosure, Denial of Service, Elevation of Privilege) e prescrevendo uma
    contramedida técnica para cada uma.
    """
)

st.divider()

uploaded_file = st.file_uploader(
    "Envie um diagrama de arquitetura (Azure, AWS ou GCP) para análise STRIDE:",
    type=["png", "jpg", "jpeg"],
    help="Formatos aceitos: PNG, JPG, JPEG.",
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption=f"Diagrama enviado: {uploaded_file.name}",
        width="stretch",
    )

    if st.button("🔍 Analisar diagrama", type="primary"):
        try:
            # O UploadedFile é um stream: cada função abaixo lê seu conteúdo, então
            # é preciso rebobinar (seek(0)) antes de cada leitura para a segunda
            # chamada não receber um stream já consumido.
            with st.spinner("Detectando componentes do diagrama (YOLO)..."):
                uploaded_file.seek(0)
                trust_boundaries, components = detect(uploaded_file)

            with st.spinner("Lendo rótulos das zonas de confiança (OCR)..."):
                uploaded_file.seek(0)
                trust_boundaries = extract_text(uploaded_file, trust_boundaries)

            with st.spinner("Associando rótulos aos componentes (OCR)..."):
                uploaded_file.seek(0)
                components = assign_component_names(
                    uploaded_file, components, trust_boundaries
                )

            with st.spinner("Montando o grafo hierárquico (LangGraph + LLM)..."):
                # Dimensões da imagem alimentam o teto de distância dos
                # proximity_hints (normalizado pela diagonal). Reaproveita
                # _load_image para abrir exatamente a mesma imagem do pipeline.
                uploaded_file.seek(0)
                image_size = _load_image(uploaded_file).size  # (width, height)
                graph = to_json(trust_boundaries, components, image_size=image_size)

            # O 'analyst' (gpt-5) devolve o parecer como StrideReport estruturado:
            # cada risco já traz o id do elemento afetado, o que permite ligar a
            # ameaça ao seu ponto exato no diagrama. É um modelo de reasoning
            # (pensa por dezenas de segundos), então cobrimos a espera com spinner.
            analyst = load_analyst_structured()
            messages = [
                SystemMessage(content=STRIDE_ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=build_stride_user_message(graph)),
            ]
            with st.spinner("O modelo está analisando o diagrama (STRIDE)..."):
                report = analyst.invoke(messages)

            # Consolida os riscos por elemento arquitetural: um bloco por
            # componente/fluxo/zona (não por ameaça), ordenado pela pior
            # severidade. Reduz drasticamente a repetição de recortes idênticos.
            groups = group_risks(report.risks)

            st.divider()
            st.subheader("📋 Parecer STRIDE")

            if not groups:
                st.info("Nenhum risco STRIDE foi identificado para este diagrama.")
            else:
                # Visão de conjunto: cada elemento com risco numerado sobre o
                # diagrama (um #N por bloco), para localizar onde intervir.
                uploaded_file.seek(0)
                st.image(
                    draw_numbered_overlay(uploaded_file, groups, graph),
                    caption="Mapa de riscos — cada número marca o elemento do bloco correspondente.",
                    width="stretch",
                )

                # Um bloco por elemento: imagem padronizada (600x400) + tabela
                # STRIDE consolidada (uma linha por ameaça daquele elemento).
                for n, group in enumerate(groups, start=1):
                    rep = group.representative
                    st.divider()
                    st.markdown(f"### #{n} — {rep.elemento_afetado} · {rep.severidade}")
                    st.markdown(
                        f"**Elemento:** {rep.elemento_afetado}  \n"
                        f"**Localização:** {_element_location(group, graph)}"
                    )
                    col_img, col_tab = st.columns([1, 2])

                    with col_img:
                        uploaded_file.seek(0)
                        crop = highlight_element(uploaded_file, group, graph)
                        if crop is not None:
                            st.image(crop, caption=f"#{n}: {rep.elemento_afetado}")
                        else:
                            st.caption(
                                f"Elemento: {rep.elemento_afetado} "
                                f"(sem localização visual disponível)."
                            )

                    with col_tab:
                        # Tabela consolidada de 4 colunas: uma linha por ameaça
                        # STRIDE do elemento (Elemento/Localização já no cabeçalho).
                        st.table(
                            {
                                "Categoria STRIDE": [r.stride_category for r in group.risks],
                                "Justificativa": [r.justificativa for r in group.risks],
                                "Severidade": [r.severidade for r in group.risks],
                                "Contramedida": [r.contramedida for r in group.risks],
                            }
                        )

            # Seção visual complementar: todas as detecções do YOLO (contexto).
            st.divider()
            st.subheader("🖼️ Componentes detectados")
            uploaded_file.seek(0)
            overlay = draw_overlay(
                uploaded_file,
                trust_boundaries,
                components,
                proximity_lines=_proximity_lines(graph),
            )
            st.image(
                overlay,
                caption="Detecções do modelo de visão computacional (YOLO).",
                width="stretch",
            )

            legend = legend_items(trust_boundaries, components)
            if legend:
                badges = "&nbsp;&nbsp;".join(
                    f"<span style='color:{hex_color}; font-size:1.3em'>■</span> {cls}"
                    for cls, hex_color in legend
                )
                st.markdown(f"**Legenda:** {badges}", unsafe_allow_html=True)

        except RuntimeError as e:
            # Ex.: OPENAI_API_KEY ausente (erro claro levantado por load_llm).
            st.error(f"Configuração ausente: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Falha ao processar o diagrama: {e}")
            st.stop()

        with st.expander("Ver JSON do grafo (dados intermediários)"):
            st.json(graph)
else:
    st.info("Envie uma imagem de diagrama de arquitetura para começar a análise.")

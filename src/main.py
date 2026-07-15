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
import time
from concurrent.futures import ThreadPoolExecutor
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

from src import branding, theme
from src.graph_builder import to_json
from src.ocr_engine import assign_component_names, extract_text
from src.pdf_report import build_pdf_report
from src.progress import (
    STEP_KEYS,
    STEP_LABELS,
    StepEstimator,
    format_elapsed,
    load_timings,
    save_timing,
)
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

# Aplica a identidade visual da marca (CSS + fontes) logo após o page_config —
# recolore os widgets nativos e habilita as classes de hero/rodapé/pill usadas
# adiante. Ver src/branding.py.
branding.inject_brand_css()


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


def _rewound(stream):
    """Rebobina o UploadedFile (seek 0) e o devolve — pronto para nova leitura.

    O stream do upload é consumido a cada leitura; o PDF chama highlight_element
    uma vez por grupo, então cada chamada precisa rebobinar antes de abrir a
    imagem de novo (mesmo cuidado dos _detect/_ocr_* do pipeline).
    """
    stream.seek(0)
    return stream


def _risk_table_html(group) -> str:
    """Tabela HTML dos riscos de um grupo, com pill de severidade por linha.

    st.table não estiliza por linha, então montamos uma tabela HTML embrulhada em
    <div class="risk-table"> — o estilo (th uppercase, bordas, cores) vem do CSS
    injetado por branding.inject_brand_css, então aqui não há estilo inline. A
    coluna Severidade usa a pill arredondada com dot do guia
    (branding.severity_pill_html), comunicando a gravidade antes da leitura. As 4
    colunas são as da consolidação (Categoria/Justificativa/Severidade/
    Contramedida).
    """
    head = (
        "<tr><th>Categoria STRIDE</th><th>Justificativa</th>"
        "<th>Severidade</th><th>Contramedida</th></tr>"
    )
    rows = []
    for r in group.risks:
        rows.append(
            "<tr>"
            f"<td>{r.stride_category}</td>"
            f"<td>{r.justificativa}</td>"
            f"<td>{branding.severity_pill_html(r.severidade)}</td>"
            f"<td>{r.contramedida}</td>"
            "</tr>"
        )
    return f"<div class='risk-table'><table>{head}{''.join(rows)}</table></div>"


def _render_log(placeholder, done_steps: list[str]) -> None:
    """Renderiza o log acumulado de etapas concluídas no placeholder."""
    if done_steps:
        placeholder.markdown("\n".join(f"✓ {label}" for label in done_steps))


def _run_step(bar, log_ph, done: list[str], estimator: StepEstimator, step: str, fn):
    """Executa uma etapa rápida (inline), medindo o tempo e atualizando a barra.

    Mostra a etapa em andamento (fração inicial da fatia), roda fn(), mede a
    duração para calibrar o ETA (save_timing), avança a barra até o fim da fatia
    da etapa e registra "✓ concluída" no log. Retorna o valor de fn().
    """
    label = STEP_LABELS[step]
    number = STEP_KEYS.index(step) + 1
    bar.progress(estimator.step_start(step), text=f"{number}. {label}…")
    t0 = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - t0
    save_timing(step, elapsed)
    bar.progress(estimator.step_end(step), text=f"{number}. {label}…")
    done.append(f"{number}. {label}")
    _render_log(log_ph, done)
    return result


branding.render_hero()

st.markdown(
    """
    Nos bastidores, a ferramenta combina três etapas: **visão computacional**
    (YOLO) para detectar os componentes do diagrama, **OCR** para ler os rótulos
    das zonas de confiança e um **LLM especialista** que atua como arquiteto
    DevSecOps — mapeando ameaças (Spoofing, Tampering, Repudiation, Information
    Disclosure, Denial of Service, Elevation of Privilege) e prescrevendo uma
    contramedida técnica para cada uma.
    """
)

uploaded_file = st.file_uploader(
    "Diagrama de arquitetura",
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
            # Barra de progresso com ETA inferido dos tempos médios de cada etapa
            # (ver src/progress.py). Dá feedback contínuo mesmo na longa etapa do
            # LLM, onde uma barra ingênua congelaria.
            estimator = StepEstimator(load_timings())
            bar = st.progress(0.0, text="Iniciando análise…")
            log_ph = st.empty()
            done: list[str] = []

            # O UploadedFile é um stream: cada função abaixo lê seu conteúdo, então
            # é preciso rebobinar (seek(0)) antes de cada leitura para a segunda
            # chamada não receber um stream já consumido.
            def _detect():
                uploaded_file.seek(0)
                return detect(uploaded_file)

            trust_boundaries, components = _run_step(
                bar, log_ph, done, estimator, "detect", _detect
            )

            def _ocr_zones():
                uploaded_file.seek(0)
                return extract_text(uploaded_file, trust_boundaries)

            trust_boundaries = _run_step(
                bar, log_ph, done, estimator, "ocr_zones", _ocr_zones
            )

            def _ocr_components():
                uploaded_file.seek(0)
                return assign_component_names(uploaded_file, components, trust_boundaries)

            components = _run_step(
                bar, log_ph, done, estimator, "ocr_components", _ocr_components
            )

            def _graph():
                # Dimensões da imagem alimentam o teto de distância dos
                # proximity_hints (normalizado pela diagonal). Reaproveita
                # _load_image para abrir exatamente a mesma imagem do pipeline.
                uploaded_file.seek(0)
                image_size = _load_image(uploaded_file).size  # (width, height)
                return to_json(trust_boundaries, components, image_size=image_size)

            graph = _run_step(bar, log_ph, done, estimator, "graph", _graph)

            # O 'analyst' (gpt-5) devolve o parecer como StrideReport estruturado:
            # cada risco já traz o id do elemento afetado, o que permite ligar a
            # ameaça ao seu ponto exato no diagrama. É um modelo de reasoning que
            # leva minutos — chamada BLOQUEANTE. Para a barra não congelar, roda
            # numa thread e animamos o progresso contra o ETA enquanto esperamos.
            analyst = load_analyst_structured()
            messages = [
                SystemMessage(content=STRIDE_ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=build_stride_user_message(graph)),
            ]
            llm_label = STEP_LABELS["llm"]
            llm_number = STEP_KEYS.index("llm") + 1
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(analyst.invoke, messages)
                while not future.done():
                    elapsed = time.monotonic() - t0
                    # Cronômetro + percentual estimado (elapsed / tempo médio do
                    # LLM) dão "sinal de vida" durante a espera longa; ficam só na
                    # barra, não no log de etapas.
                    percent = estimator.step_percent("llm", elapsed)
                    bar.progress(
                        estimator.fraction("llm", elapsed),
                        text=f"{llm_number}. {llm_label}… {percent}% ({format_elapsed(elapsed)})",
                    )
                    time.sleep(0.2)
                report = future.result()  # propaga exceção da thread ao except

            save_timing("llm", time.monotonic() - t0)
            bar.progress(1.0, text="Concluído")
            done.append(f"{llm_number}. {llm_label}")
            _render_log(log_ph, done)

            # Consolida os riscos por elemento arquitetural: um bloco por
            # componente/fluxo/zona (não por ameaça), ordenado pela pior
            # severidade. Reduz drasticamente a repetição de recortes idênticos.
            groups = group_risks(report.risks)

            st.divider()
            st.subheader("📋 Parecer STRIDE")

            if not groups:
                st.info("Nenhum risco STRIDE foi identificado para este diagrama.")
            else:
                # Overlay numerado reusado tanto na tela quanto no PDF.
                uploaded_file.seek(0)
                numbered_overlay = draw_numbered_overlay(uploaded_file, groups, graph)

                # Exportação em PDF: gerado a partir dos MESMOS dados já
                # calculados (groups/graph/overlay), reaproveitando _element_location
                # e highlight_element via funções injetadas. Botão logo no topo do
                # parecer, para baixar sem rolar a página inteira.
                pdf_bytes = build_pdf_report(
                    diagram_name=uploaded_file.name,
                    groups=groups,
                    graph=graph,
                    numbered_overlay=numbered_overlay,
                    element_location_fn=_element_location,
                    card_image_fn=lambda g, gr: highlight_element(
                        _rewound(uploaded_file), g, gr
                    ),
                )
                st.download_button(
                    "⬇️ Baixar relatório em PDF",
                    data=pdf_bytes,
                    file_name=f"stride_report_{uploaded_file.name.rsplit('.', 1)[0]}.pdf",
                    mime="application/pdf",
                    type="primary",
                )

                # Visão de conjunto: cada elemento com risco numerado sobre o
                # diagrama (um #N por bloco), para localizar onde intervir.
                st.image(
                    numbered_overlay,
                    caption="Mapa de riscos — cada número marca o elemento do bloco correspondente.",
                    width="stretch",
                )

                # Um bloco por elemento: imagem padronizada (600x400) + tabela
                # STRIDE consolidada (uma linha por ameaça daquele elemento). Cada
                # bloco vive num st.container(border=True), estilizado como card
                # (navy) pelo CSS da marca — dispensa o st.divider entre blocos.
                for n, group in enumerate(groups, start=1):
                    rep = group.representative
                    with st.container(border=True):
                        # Cabeçalho com um marcador na cor da pior severidade do
                        # bloco, para a hierarquia por gravidade ler de relance.
                        sev_color = theme.severity_color(rep.severidade)
                        st.markdown(
                            f"### <span style='color:{sev_color}'>■</span> "
                            f"#{n} — {rep.elemento_afetado} · {rep.severidade}",
                            unsafe_allow_html=True,
                        )
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
                            # STRIDE do elemento, severidade como pill colorida.
                            st.markdown(_risk_table_html(group), unsafe_allow_html=True)

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
    st.info(
        "Envie um diagrama de arquitetura cloud (AWS, Azure ou GCP) e receba uma "
        "análise de segurança automatizada pela metodologia STRIDE."
    )

# Rodapé de marca no lugar do 'Made with Streamlit' oculto — selo + slogan.
branding.render_footer()

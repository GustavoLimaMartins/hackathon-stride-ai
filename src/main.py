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
from src.stride_engine import load_llm
from src.vision import detect
from src.visual_report import draw_overlay, legend_items

# set_page_config deve ser a primeira chamada Streamlit do script.
st.set_page_config(
    page_title="STRIDE-AI — Análise de Ameaças em Diagramas de Arquitetura Cloud",
    page_icon="🛡️",
    layout="wide",
)

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
                graph = to_json(trust_boundaries, components)

            llm = load_llm("analyst")
            messages = [
                SystemMessage(content=STRIDE_ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=build_stride_user_message(graph)),
            ]

            # O 'analyst' (gpt-5) é um modelo de reasoning: ele "pensa" por dezenas
            # de segundos antes de emitir o primeiro token. O spinner cobre essa
            # espera inicial; assim que o primeiro pedaço de texto chega, cedemos
            # lugar ao streaming token a token (que já é o indicador de progresso).
            token_stream = llm.stream(messages)
            chunks: list[str] = []

            with st.spinner("O modelo está analisando o diagrama (STRIDE)..."):
                first_chunk = ""
                for chunk in token_stream:
                    if chunk.content:
                        first_chunk = chunk.content
                        chunks.append(first_chunk)
                        break

            st.divider()
            st.subheader("📋 Parecer STRIDE")

            def _stream_response():
                # Reemite o primeiro chunk já consumido (fora do spinner) e segue
                # com o restante do stream. Acumula em 'chunks' para ter a resposta
                # completa garantida ao final, independente do retorno do widget.
                if first_chunk:
                    yield first_chunk
                for chunk in token_stream:
                    chunks.append(chunk.content)
                    yield chunk.content

            st.write_stream(_stream_response)
            full_response = "".join(chunks)

            # Seção visual: as caixas detectadas plotadas sobre a imagem enviada,
            # dando a visão espacial de "o que o modelo viu" ao lado do parecer.
            st.divider()
            st.subheader("🖼️ Componentes detectados")
            uploaded_file.seek(0)
            overlay = draw_overlay(uploaded_file, trust_boundaries, components)
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

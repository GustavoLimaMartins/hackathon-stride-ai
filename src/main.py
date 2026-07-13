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

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph_builder import to_json
from src.ocr_engine import extract_text
from src.prompts import STRIDE_ANALYST_SYSTEM_PROMPT, build_stride_user_message
from src.stride_engine import load_llm
from src.vision import detect

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
        use_container_width=True,
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

            with st.spinner("Montando o grafo hierárquico (LangGraph + LLM)..."):
                graph = to_json(trust_boundaries, components)

            with st.spinner(
                "Gerando parecer STRIDE (isso pode levar até um minuto)..."
            ):
                llm = load_llm("analyst")
                messages = [
                    SystemMessage(content=STRIDE_ANALYST_SYSTEM_PROMPT),
                    HumanMessage(content=build_stride_user_message(graph)),
                ]
                response = llm.invoke(messages)

        except RuntimeError as e:
            # Ex.: OPENAI_API_KEY ausente (erro claro levantado por load_llm).
            st.error(f"Configuração ausente: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Falha ao processar o diagrama: {e}")
            st.stop()

        st.divider()
        st.subheader("📋 Parecer STRIDE")
        st.markdown(response.content)

        with st.expander("Ver JSON do grafo (dados intermediários)"):
            st.json(graph)
else:
    st.info("Envie uma imagem de diagrama de arquitetura para começar a análise.")

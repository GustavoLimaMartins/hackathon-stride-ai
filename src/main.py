"""Frontend Streamlit: a vitrine da análise STRIDE de diagramas de arquitetura.

Esta é a camada de apresentação do projeto. O backend (Fases 2-4) já entrega o
pipeline completo: visão computacional (YOLO) detecta os componentes do diagrama,
OCR lê os rótulos das zonas de confiança, a engenharia espacial monta um grafo
hierárquico em JSON e o LLM ('analyst') produz o parecer STRIDE com contramedidas.

Microtarefa 5.1: apenas o esqueleto da interface — título, descrição e upload da
imagem com pré-visualização. A orquestração do pipeline ao enviar a imagem entra
em uma microtarefa seguinte (5.2).

Execução:
    streamlit run src/main.py
"""

import streamlit as st

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
else:
    st.info("Envie uma imagem de diagrama de arquitetura para começar a análise.")

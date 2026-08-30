import os

import streamlit as st

# Carrega os Secrets configurados no Streamlit Cloud.
try:
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_BASE_URL",
    ):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except FileNotFoundError:
    pass

from src.ingestion import index_documents
from src.rag import answer

st.set_page_config(
    page_title="PI-5 • RAG",
    page_icon="📚",
)

st.title("📚 Chatbot PI-5 — Fundeb, VAAR e ICMS Educacional")
st.caption("Respostas ancoradas na documentação local indexada.")


@st.cache_resource(
    show_spinner="Preparando e indexando os documentos pela primeira vez..."
)
def ensure_index():
    # O upsert evita duplicação dos documentos.
    return index_documents(reset=False)


try:
    ensure_index()
except Exception as error:
    st.error(f"Não foi possível preparar a base documental: {error}")
    st.stop()


question = st.chat_input("Faça uma pergunta sobre os documentos do PI-5")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando os documentos..."):
            try:
                response, sources = answer(question)
                st.write(response)

                with st.expander("Trechos recuperados"):
                    for item in sources:
                        st.markdown(
                            f"**{item['source']} — página {item['page']}**"
                        )
                        st.write(item["text"])
            except Exception as error:
                st.error(str(error))

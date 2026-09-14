import os

import streamlit as st


# =========================================================
# SECRETS / VARIÁVEIS DE AMBIENTE
# =========================================================

# Carrega os Secrets configurados no Streamlit Cloud.
try:
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_BASE_URL",
    ):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except (FileNotFoundError, KeyError):
    pass


from src.ingestion import index_documents
from src.rag import answer


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="PI-5 • RAG",
    page_icon="📚",
    layout="centered",
)


st.title("📚 Chatbot PI-5 — Fundeb, VAAR e ICMS Educacional")

st.caption(
    "Respostas baseadas exclusivamente na documentação local indexada."
)


# =========================================================
# INDEXAÇÃO
# =========================================================

@st.cache_resource(
    show_spinner="Preparando e indexando os documentos..."
)
def ensure_index():
    """
    Reconstrói o índice uma vez por inicialização da aplicação.

    O cache do Streamlit impede que isso seja executado
    novamente a cada pergunta.
    """
@st.cache_resource(
    show_spinner="Preparando e indexando os documentos..."
)
def ensure_index():
    return index_documents(
        reset=False
    )

try:
    total_chunks = ensure_index()

except Exception as error:
    st.error(
        "Não foi possível preparar a base documental."
    )

    st.code(str(error))

    st.stop()


# =========================================================
# ESTADO DA CONVERSA
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# FUNÇÃO PARA EXIBIR FONTES
# =========================================================

def format_source_label(item: dict) -> str:
    """
    Formata a identificação da fonte conforme o tipo
    de documento recuperado.
    """

    # Se o rag.py já gerou a citação, usamos diretamente.
    citation = item.get("citation")

    if citation:
        return citation

    source = item.get(
        "source",
        "Fonte desconhecida",
    )

    document_type = item.get(
        "type",
        "document",
    )

    # -------------------------
    # CSV
    # -------------------------

    if document_type == "csv":

        entity = item.get("entity")
        row = item.get("row")

        if entity and row:
            return (
                f"{source} — "
                f"{entity} — linha {row}"
            )

        if entity:
            return f"{source} — {entity}"

        if row:
            return f"{source} — linha {row}"

        return source

    # -------------------------
    # PDF / TXT
    # -------------------------

    page = item.get("page")

    if page:
        return f"{source} — página {page}"

    return source


# =========================================================
# MOSTRA HISTÓRICO
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources")

        if sources:

            with st.expander(
                f"📄 Trechos recuperados ({len(sources)})"
            ):

                for index, item in enumerate(
                    sources,
                    start=1,
                ):

                    source_label = format_source_label(
                        item
                    )

                    st.markdown(
                        f"**{index}. {source_label}**"
                    )

                    st.write(
                        item.get(
                            "text",
                            "Trecho indisponível.",
                        )
                    )

                    if index < len(sources):
                        st.divider()


# =========================================================
# ENTRADA DO USUÁRIO
# =========================================================

question = st.chat_input(
    "Faça uma pergunta sobre Fundeb, VAAR, SAERS, IMERS, PRE ou ICMS Educacional"
)


if question:

    # Guarda a pergunta no histórico
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # =====================================================
    # RESPOSTA
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Consultando os documentos..."
        ):

            try:

                response, sources = answer(
                    question
                )

                st.markdown(response)


                # -----------------------------------------
                # FONTES
                # -----------------------------------------

                if sources:

                    with st.expander(
                        f"📄 Trechos recuperados ({len(sources)})"
                    ):

                        for index, item in enumerate(
                            sources,
                            start=1,
                        ):

                            source_label = (
                                format_source_label(
                                    item
                                )
                            )

                            st.markdown(
                                f"**{index}. "
                                f"{source_label}**"
                            )

                            st.write(
                                item.get(
                                    "text",
                                    "Trecho indisponível.",
                                )
                            )

                            if index < len(sources):
                                st.divider()


                # Guarda resposta + fontes
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    }
                )


            except Exception as error:

                error_message = str(error)

                st.error(
                    "Não foi possível gerar a resposta."
                )

                st.code(
                    error_message
                )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Base documental")

    st.write(
        "Documentos jurídicos e dados utilizados "
        "pelo chatbot."
    )

    st.metric(
        "Chunks indexados",
        total_chunks,
    )

    st.markdown(
        """
**Formatos suportados**

- 📄 PDF
- 📝 TXT
- 📊 CSV
"""
    )

    st.divider()

    st.markdown(
        """
**Principais temas**

- Fundeb
- VAAR
- SAERS
- IMERS
- PRE
- ICMS Educacional
"""
    )

    st.divider()

    if st.button(
        "🗑️ Limpar conversa",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

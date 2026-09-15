import os

import streamlit as st


# =========================================================
# SECRETS -> ENV
# =========================================================

try:
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_BASE_URL",
        "EMBEDDING_MODEL",
        "TOP_K",
        "CANDIDATE_K",
        "MAX_CONTEXT_CHUNKS",
    ):
        if key in st.secrets:
            os.environ[key] = str(
                st.secrets[key]
            )
except Exception:
    pass


from src.ingestion import index_documents
from src.rag import (
    answer,
    clear_rag_caches,
)


st.set_page_config(
    page_title=(
        "Chatbot PI-5 — Fundeb, VAAR "
        "e ICMS Educacional"
    ),
    page_icon="📚",
    layout="wide",
)


st.title(
    "📚 Chatbot PI-5 — Fundeb, VAAR "
    "e ICMS Educacional"
)

st.caption(
    "Respostas baseadas exclusivamente "
    "na documentação local indexada."
)


# =========================================================
# INDEXAÇÃO
# =========================================================

@st.cache_resource(
    show_spinner=False
)
def ensure_index():
    return index_documents(
        reset=False
    )


try:
    with st.spinner(
        "Preparando e indexando os documentos..."
    ):
        indexed_chunks = ensure_index()

except Exception as error:
    st.error(
        "Não foi possível preparar "
        "a base documental."
    )
    st.code(
        str(error)
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.subheader(
        "Base documental"
    )

    st.metric(
        "Chunks indexados",
        indexed_chunks,
    )

    st.caption(
        "PDF/TXT são priorizados em perguntas "
        "conceituais. CSV é usado para perguntas "
        "sobre municípios, valores e indicadores."
    )

    if st.button(
        "🔄 Reconstruir índice",
        use_container_width=True,
    ):
        try:
            clear_rag_caches()
            st.cache_resource.clear()

            with st.spinner(
                "Reconstruindo índice..."
            ):
                total = index_documents(
                    reset=True
                )

            st.success(
                f"Índice reconstruído com "
                f"{total} chunks."
            )

            st.rerun()

        except Exception as error:
            st.error(
                "Falha ao reconstruir o índice."
            )
            st.code(
                str(error)
            )

    if st.button(
        "🧹 Limpar conversa",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# CHAT
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


def source_label(
    item: dict,
) -> str:

    source = item.get(
        "source",
        "Documento",
    )

    doc_type = item.get(
        "type",
        "document",
    )

    if doc_type == "csv":
        details: list[str] = []

        entity = item.get("entity")
        row = item.get("row")

        if entity:
            details.append(
                f"município: {entity}"
            )

        if row:
            details.append(
                f"linha {row}"
            )

        if details:
            return (
                f"{source} — "
                f"{', '.join(details)}"
            )

        return source

    if doc_type == "pdf":
        page = item.get("page")

        if page:
            return (
                f"{source} — página {page}"
            )

    return source


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and message.get("sources")
        ):
            with st.expander(
                "📄 Trechos recuperados"
            ):
                for index, item in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"**{index}. "
                        f"{source_label(item)}**"
                    )

                    st.write(
                        item.get(
                            "text",
                            "",
                        )
                    )


question = st.chat_input(
    "Digite sua pergunta sobre Fundeb, "
    "VAAR, SAERS, IMERS ou ICMS Educacional..."
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message(
        "assistant"
    ):
        try:
            with st.spinner(
                "Consultando a documentação..."
            ):
                response, sources = answer(
                    question
                )

            st.markdown(response)

            if sources:
                with st.expander(
                    f"📄 Trechos recuperados "
                    f"({len(sources)})"
                ):
                    for index, item in enumerate(
                        sources,
                        start=1,
                    ):
                        st.markdown(
                            f"**{index}. "
                            f"{source_label(item)}**"
                        )

                        st.write(
                            item.get(
                                "text",
                                "",
                            )
                        )

        except Exception as error:
            response = (
                "Não foi possível gerar a resposta."
            )
            sources = []

            st.error(response)
            st.code(
                str(error)
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
        }
    )

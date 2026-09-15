import os
from pathlib import Path

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


# =========================================================
# IMPORTS DO PROJETO
# =========================================================

from src.ingestion import index_documents
from src.rag import (
    answer,
    clear_rag_caches,
)
from src.settings import (
    RAW_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CANDIDATE_K,
    MAX_CONTEXT_CHUNKS,
)


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Chatbot Educacional RS",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# ESTILO
# =========================================================

st.markdown(
    """
    <style>

    /* Fundo geral */
    .stApp {
        background:
            linear-gradient(
                135deg,
                #07111f 0%,
                #0c1728 55%,
                #101c30 100%
            );
    }

    /* Limite de largura */
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    /* Título principal */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #f7f9fc;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #aeb9ca;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .info-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 17px 18px;
        margin-bottom: 10px;
    }

    .info-card-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f3f6fb;
        margin-bottom: 5px;
    }

    .info-card-text {
        color: #aeb9ca;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Documento */
    .doc-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 7px;
    }

    .doc-name {
        color: #f5f7fa;
        font-weight: 600;
        font-size: 0.87rem;
        word-break: break-word;
    }

    .doc-type {
        color: #8895a8;
        font-size: 0.75rem;
        margin-top: 3px;
    }

    /* Badge online */
    .status-online {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(48, 209, 88, 0.12);
        color: #74e28f;
        border: 1px solid rgba(48,209,88,0.22);
        margin-bottom: 10px;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 6px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        background: rgba(255,255,255,0.025);
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        padding: 14px;
        border-radius: 12px;
    }

    /* Botões */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def get_available_documents() -> list[Path]:
    """
    Lista automaticamente os documentos presentes
    na pasta raw/.
    """

    if not RAW_DIR.exists():
        return []

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".csv",
        ".md",
    }

    documents = []

    for path in RAW_DIR.iterdir():
        if (
            path.is_file()
            and path.suffix.lower()
            in allowed_extensions
        ):
            documents.append(path)

    return sorted(
        documents,
        key=lambda item: item.name.lower(),
    )


def document_icon(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return "📕"

    if suffix == ".txt":
        return "📄"

    if suffix == ".csv":
        return "📊"

    if suffix == ".md":
        return "📝"

    return "📁"


def document_description(
    filename: str,
) -> str:

    name = filename.lower()

    if "saers" in name:
        return (
            "Avaliação educacional e desempenho "
            "da educação no Rio Grande do Sul."
        )

    if "imers" in name:
        return (
            "Índice Municipal da Qualidade "
            "da Educação do Rio Grande do Sul."
        )

    if "pre" in name:
        return (
            "Participação no Rateio da "
            "Cota-Parte da Educação."
        )

    if "icms" in name:
        return (
            "Legislação e regras relacionadas "
            "ao ICMS Educacional."
        )

    if "fundeb" in name:
        return (
            "Documentação relacionada ao "
            "Fundeb."
        )

    if "vaar" in name:
        return (
            "Documentação relacionada à "
            "complementação VAAR."
        )

    return "Documento disponível para consulta."


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


# =========================================================
# DOCUMENTOS
# =========================================================

available_documents = (
    get_available_documents()
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
        "Preparando base documental..."
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
# CABEÇALHO
# =========================================================

st.markdown(
    """
    <div class="main-title">
        📚 Chatbot Educacional RS
    </div>

    <div class="main-subtitle">
        Consulta inteligente à documentação sobre
        SAERS, IMERS, PRE e ICMS Educacional
        do Rio Grande do Sul.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MÉTRICAS PRINCIPAIS
# =========================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "📚 Documentos",
        len(available_documents),
    )

with metric2:
    st.metric(
        "🧩 Chunks indexados",
        indexed_chunks,
    )

with metric3:
    model_name = os.getenv(
        "OPENROUTER_MODEL",
        "Modelo não informado",
    )

    short_model = (
        model_name.split("/")[-1]
        if "/"
        in model_name
        else model_name
    )

    st.metric(
        "🤖 Modelo",
        short_model,
    )


# =========================================================
# INFORMAÇÕES
# =========================================================

with st.expander(
    "ℹ️ Sobre este chatbot"
):
    st.markdown(
        """
        Este chatbot utiliza **RAG
        (Retrieval-Augmented Generation)**.

        Quando uma pergunta é realizada:

        **1.** A pergunta é analisada pelo sistema.

        **2.** Os trechos mais relevantes da
        documentação são recuperados.

        **3.** Esses trechos são enviados ao
        modelo de linguagem.

        **4.** A resposta é construída utilizando
        a documentação recuperada como contexto.

        O objetivo é reduzir respostas inventadas
        e permitir que as informações possam ser
        verificadas diretamente nos documentos.
        """
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="
            margin-bottom:4px;
        ">
            📚 Base documental
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="status-online">
            ● Base carregada
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Documentos atualmente disponíveis "
        "para o chatbot."
    )

    st.divider()

    # =====================================================
    # DOCUMENTOS DISPONÍVEIS
    # =====================================================

    if available_documents:

        for document in available_documents:

            icon = document_icon(
                document
            )

            description = (
                document_description(
                    document.name
                )
            )

            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-name">
                        {icon} {document.name}
                    </div>

                    <div class="doc-type">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.warning(
            "Nenhum documento encontrado "
            "na pasta raw/."
        )

    st.divider()

    # =====================================================
    # COBERTURA
    # =====================================================

    st.subheader(
        "🎯 Assuntos disponíveis"
    )

    st.markdown(
        """
        - **SAERS**
        - **IMERS**
        - **PRE**
        - **ICMS Educacional**
        - Indicadores educacionais
        - Porte dos municípios
        - IPM
        """
    )

    st.divider()

    # =====================================================
    # CONFIGURAÇÃO RAG
    # =====================================================

    with st.expander(
        "⚙️ Configuração do RAG"
    ):

        st.write(
            f"**Embedding:** "
            f"`{EMBEDDING_MODEL}`"
        )

        st.write(
            f"**TOP K:** `{TOP_K}`"
        )

        st.write(
            f"**Candidatos:** "
            f"`{CANDIDATE_K}`"
        )

        st.write(
            f"**Máximo de contexto:** "
            f"`{MAX_CONTEXT_CHUNKS}`"
        )

        st.write(
            f"**Chunks indexados:** "
            f"`{indexed_chunks}`"
        )

    st.divider()

    # =====================================================
    # BOTÕES
    # =====================================================

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
                f"Índice reconstruído "
                f"com {total} chunks."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Falha ao reconstruir "
                "o índice."
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

    st.divider()

    st.caption(
        "PI-5 • RAG educacional • "
        "Rio Grande do Sul"
    )


# =========================================================
# PERGUNTAS SUGERIDAS
# =========================================================

st.markdown(
    "### 💡 Exemplos de perguntas"
)

example1, example2, example3 = (
    st.columns(3)
)

with example1:
    st.info(
        "O que é o SAERS e qual "
        "é sua finalidade?"
    )

with example2:
    st.info(
        "Como o porte do município "
        "influencia a PRE?"
    )

with example3:
    st.info(
        "Qual é a relação entre "
        "IMERS e PRE?"
    )


st.divider()


# =========================================================
# HISTÓRICO
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


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
                f"📄 Fontes consultadas "
                f"({len(message['sources'])})"
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


# =========================================================
# INPUT
# =========================================================

question = st.chat_input(
    "Pergunte sobre SAERS, IMERS, "
    "PRE ou ICMS Educacional..."
)


# =========================================================
# RESPOSTA
# =========================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

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

            st.markdown(
                response
            )

            if sources:

                with st.expander(
                    f"📄 Fontes consultadas "
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
                "Não foi possível gerar "
                "a resposta."
            )

            sources = []

            st.error(
                response
            )

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

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# DIRETÓRIOS DO PROJETO
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "raw"


# =========================================================
# CHROMADB
# =========================================================

# No Streamlit Cloud, evitamos reutilizar sempre o mesmo SQLite.
#
# Se CHROMA_DIR estiver definido no .env / Secrets, ele será usado.
# Caso contrário, uma pasta temporária NOVA é criada quando
# o servidor inicia.
#
# Exemplo:
# /tmp/pi5_chroma_a81h2k3x
#
# Isso evita erros como:
# - attempt to write a readonly database
# - no such table: tenants
# - banco SQLite antigo/incompatível

_chroma_env = os.getenv("CHROMA_DIR")

if _chroma_env:
    CHROMA_DIR = Path(_chroma_env)
    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
else:
    CHROMA_DIR = Path(
        tempfile.mkdtemp(
            prefix="pi5_chroma_"
        )
    )


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "pi5_documentos",
)


# =========================================================
# EMBEDDINGS
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


# =========================================================
# CHUNKING
# =========================================================

# Tamanho de cada trecho de documento.
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "900",
    )
)

# Quantidade de caracteres compartilhados
# entre dois chunks consecutivos.
CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "180",
    )
)

# Quantidade de chunks enviada ao Chroma
# por lote durante a indexação.
INDEX_BATCH_SIZE = int(
    os.getenv(
        "INDEX_BATCH_SIZE",
        "48",
    )
)


# =========================================================
# RECUPERAÇÃO RAG
# =========================================================

# Quantidade final de trechos que queremos utilizar.
TOP_K = int(
    os.getenv(
        "TOP_K",
        "12",
    )
)

# Busca uma quantidade maior antes do reranking.
CANDIDATE_K = int(
    os.getenv(
        "CANDIDATE_K",
        "28",
    )
)

# Limite máximo de chunks enviados como contexto ao LLM.
MAX_CONTEXT_CHUNKS = int(
    os.getenv(
        "MAX_CONTEXT_CHUNKS",
        "14",
    )
)


# =========================================================
# VERSÃO DO ÍNDICE
# =========================================================

# Alterar esta versão pode ser usado para indicar
# mudanças relevantes na estratégia de indexação.
INDEX_VERSION = os.getenv(
    "INDEX_VERSION",
    "pi5-rag-2026-09-15-v5",
)

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "raw"

# No Streamlit Cloud, usar /tmp evita problemas de permissão e bancos SQLite
# antigos/incompatíveis dentro do repositório.
CHROMA_DIR = Path(
    os.getenv(
        "CHROMA_DIR",
        str(Path(tempfile.gettempdir()) / "pi5_chroma_v4"),
    )
)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "pi5_documentos",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Estratégia de chunking:
# chunks menores aumentam a granularidade; overlap evita perder contexto
# entre o fim de um trecho e o começo do próximo.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "48"))

# Recuperação:
# TOP_K é a quantidade alvo de trechos enviados ao LLM.
# CANDIDATE_K é a quantidade maior de candidatos buscados antes do reranking.
TOP_K = int(os.getenv("TOP_K", "12"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "28"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "14"))

# Alterar esta versão força a reconstrução do índice.
INDEX_VERSION = os.getenv(
    "INDEX_VERSION",
    "pi5-rag-2026-09-14-v4",
)

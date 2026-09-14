from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


# Documentos do RAG
RAW_DIR = ROOT / "raw"


# Banco vetorial Chroma
#
# No Streamlit Cloud, /tmp é gravável.
# No Windows/local, tempfile.gettempdir() também funciona normalmente.
DEFAULT_CHROMA_DIR = (
    Path(tempfile.gettempdir())
    / "pi5_chroma"
)

CHROMA_DIR = Path(
    os.getenv(
        "CHROMA_DIR",
        str(
            Path(tempfile.gettempdir())
            / "pi5_chroma_v2"
        ),
    )
)

# Garante que a pasta exista antes do SQLite/Chroma abrir o banco.
CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COLLECTION_NAME = "pi5_documentos"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "5",
    )
)

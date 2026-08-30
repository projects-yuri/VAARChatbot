from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "pi5_documentos"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
TOP_K = int(os.getenv("TOP_K", "5"))

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

from src.settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RAW_DIR

LEGAL_BOUNDARY = re.compile(
    r"(?=^\s*(?:Art\.\s*\d+[ºo°]?|§\s*\d+[ºo°]?|Parágrafo\s+(?:único|\d+)|[IVXLCDM]+\s*[-–]|[a-z]\)\s*))",
    re.IGNORECASE | re.MULTILINE,
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")


def clean_text(text: str) -> str:
    """Normaliza espaços sem alterar títulos, artigos ou numeração jurídica."""
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> Iterable[tuple[str, int]]:
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            yield text, page_number


def split_legal_text(text: str, chunk_size: int = 1_300, overlap: int = 180) -> list[str]:
    """Prioriza Art., §, incisos e alíneas; só divide por frase blocos longos."""
    units = [clean_text(unit) for unit in LEGAL_BOUNDARY.split(text) if clean_text(unit)]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_large_unit(unit, chunk_size, overlap))
        elif not current:
            current = unit
        elif len(current) + len(unit) + 2 <= chunk_size:
            current += "\n\n" + unit
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def _split_large_unit(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = SENTENCE_BOUNDARY.split(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current or len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue
        chunks.append(current)
        current = (current[-overlap:] + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks


def document_records(raw_dir: Path) -> Iterable[tuple[str, dict]]:
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = extract_pdf(path)
        elif suffix == ".txt":
            pages = [(clean_text(path.read_text(encoding="utf-8", errors="replace")), 1)]
        else:
            continue
        for page_text, page in pages:
            for index, chunk in enumerate(split_legal_text(page_text)):
                if len(chunk) >= 40:
                    yield chunk, {"source": path.name, "page": page, "chunk": index}


def index_documents(raw_dir: Path = RAW_DIR, reset: bool = False) -> int:
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(f"Nenhum documento encontrado em: {raw_dir}")
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding)

    documents, metadatas, ids = [], [], []
    for text, metadata in document_records(raw_dir):
        stable_id = hashlib.sha256(
            f"{metadata['source']}|{metadata['page']}|{metadata['chunk']}|{text}".encode()
        ).hexdigest()
        documents.append(text)
        metadatas.append(metadata)
        ids.append(stable_id)
    if not documents:
        raise ValueError("Não foi possível extrair texto dos documentos.")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa os documentos do PI-5 no Chroma local.")
    parser.add_argument("--reset", action="store_true", help="Apaga o índice local antes de recriá-lo.")
    args = parser.parse_args()
    total = index_documents(reset=args.reset)
    print(f"Índice criado/atualizado com {total} chunks em {CHROMA_DIR}")

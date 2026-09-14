from __future__ import annotations

import argparse
import csv
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


# =========================================================
# PDF
# =========================================================

def extract_pdf(path: Path) -> Iterable[tuple[str, int]]:
    reader = PdfReader(str(path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if text:
            yield text, page_number


# =========================================================
# CSV
# =========================================================

def detect_csv_encoding(path: Path) -> str:
    """
    Tenta detectar uma codificação compatível com os CSVs oficiais.
    Alguns arquivos do FNDE/MEC usam Latin-1/Windows-1252.
    """
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return "latin-1"


def normalize_column_name(value: str) -> str:
    """Remove quebras de linha e espaços extras dos nomes das colunas."""
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def find_csv_header(rows: list[list[str]]) -> int:
    """
    Localiza a linha de cabeçalho.

    Os CSVs oficiais do VAAR possuem algumas linhas de título antes
    da tabela. Procuramos primeiro por 'UF' + 'Ente Federado'.
    """
    for index, row in enumerate(rows):
        normalized = [normalize_column_name(cell).lower() for cell in row]

        if "uf" in normalized and "ente federado" in normalized:
            return index

    # Fallback: primeira linha com pelo menos duas células preenchidas
    for index, row in enumerate(rows):
        filled = [cell for cell in row if cell.strip()]

        if len(filled) >= 2:
            return index

    return 0


def extract_csv(path: Path) -> Iterable[tuple[str, dict]]:
    """
    Converte cada linha do CSV em um documento textual independente.

    Isso é melhor para busca vetorial do que indexar a tabela inteira
    em um único chunk.
    """
    encoding = detect_csv_encoding(path)

    raw_text = path.read_text(
        encoding=encoding,
        errors="replace",
    )

    try:
        dialect = csv.Sniffer().sniff(
            raw_text[:20_000],
            delimiters=";,\t|",
        )
        delimiter = dialect.delimiter
    except csv.Error:
        # Os arquivos oficiais utilizados no projeto usam ;
        delimiter = ";"

    rows = list(
        csv.reader(
            raw_text.splitlines(),
            delimiter=delimiter,
        )
    )

    if not rows:
        return

    header_index = find_csv_header(rows)

    headers = [
        normalize_column_name(column)
        for column in rows[header_index]
    ]

    for row_number, row in enumerate(
        rows[header_index + 1:],
        start=header_index + 2,
    ):
        if not any(cell.strip() for cell in row):
            continue

        # completa linha caso tenha menos células que o cabeçalho
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))

        data = {}

        for header, value in zip(headers, row):
            header = clean_text(header)
            value = clean_text(value)

            if header and value:
                data[header] = value

        if not data:
            continue

        # Identificadores úteis para recuperação
        uf = data.get("UF", "")
        ente = data.get("Ente Federado", "")
        codigo_ibge = (
            data.get("Código IBGE")
            or data.get("Código IBGE ")
            or data.get("Código IBGE", "")
        )

        # Contexto semântico adicional.
        # Ajuda o embedding a entender que os dados são sobre VAAR.
        parts = [
            "Dados oficiais do VAAR 2026.",
            "Complementação da União ao Fundeb - VAAR.",
        ]

        if ente:
            parts.append(f"Ente Federado: {ente}.")

        if uf:
            parts.append(f"UF: {uf}.")

        if codigo_ibge:
            parts.append(f"Código IBGE: {codigo_ibge}.")

        # Adiciona todas as colunas da linha
        for column, value in data.items():
            if column in {"UF", "Ente Federado", "Código IBGE"}:
                continue

            parts.append(f"{column}: {value}.")

        text = "\n".join(parts)

        metadata = {
            "source": path.name,
            "type": "csv",
            "row": row_number,
            # Mantemos page para não quebrar o rag.py atual
            "page": 1,
            "chunk": 0,
        }

        if uf:
            metadata["uf"] = uf

        if ente:
            metadata["entity"] = ente

        if codigo_ibge:
            metadata["codigo_ibge"] = str(codigo_ibge)

        yield text, metadata


# =========================================================
# CHUNKING DE TEXTOS JURÍDICOS
# =========================================================

def split_legal_text(
    text: str,
    chunk_size: int = 1_300,
    overlap: int = 180,
) -> list[str]:
    """
    Prioriza Art., §, incisos e alíneas;
    só divide por frase blocos longos.
    """
    units = [
        clean_text(unit)
        for unit in LEGAL_BOUNDARY.split(text)
        if clean_text(unit)
    ]

    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > chunk_size:

            if current:
                chunks.append(current)
                current = ""

            chunks.extend(
                _split_large_unit(
                    unit,
                    chunk_size,
                    overlap,
                )
            )

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


def _split_large_unit(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    sentences = SENTENCE_BOUNDARY.split(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:

        if not current or len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue

        chunks.append(current)

        current = (
            current[-overlap:] + " " + sentence
        ).strip()

    if current:
        chunks.append(current)

    return chunks


# =========================================================
# DOCUMENTOS
# =========================================================

def document_records(
    raw_dir: Path,
) -> Iterable[tuple[str, dict]]:

    for path in sorted(raw_dir.iterdir()):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        # -------------------------
        # PDF
        # -------------------------

        if suffix == ".pdf":

            for page_text, page in extract_pdf(path):

                for index, chunk in enumerate(
                    split_legal_text(page_text)
                ):

                    if len(chunk) >= 40:

                        yield chunk, {
                            "source": path.name,
                            "type": "pdf",
                            "page": page,
                            "chunk": index,
                        }

        # -------------------------
        # TXT
        # -------------------------

        elif suffix == ".txt":

            text = clean_text(
                path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            for index, chunk in enumerate(
                split_legal_text(text)
            ):

                if len(chunk) >= 40:

                    yield chunk, {
                        "source": path.name,
                        "type": "txt",
                        "page": 1,
                        "chunk": index,
                    }

        # -------------------------
        # CSV
        # -------------------------

        elif suffix == ".csv":

            for text, metadata in extract_csv(path):

                if len(text) >= 20:
                    yield text, metadata


# =========================================================
# INDEXAÇÃO
# =========================================================

def index_documents(
    raw_dir: Path = RAW_DIR,
    reset: bool = False,
) -> int:

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(
            f"Nenhum documento encontrado em: {raw_dir}"
        )

if reset and CHROMA_DIR.exists():
    shutil.rmtree(
        CHROMA_DIR,
        ignore_errors=True,
    )

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

    embedding = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding,
    )

    documents = []
    metadatas = []
    ids = []

    for text, metadata in document_records(raw_dir):

        locator = (
            metadata.get("row")
            or metadata.get("page")
            or 0
        )

        stable_id = hashlib.sha256(
            (
                f"{metadata['source']}|"
                f"{metadata.get('type', 'document')}|"
                f"{locator}|"
                f"{metadata.get('chunk', 0)}|"
                f"{text}"
            ).encode("utf-8")
        ).hexdigest()

        documents.append(text)
        metadatas.append(metadata)
        ids.append(stable_id)

    if not documents:
        raise ValueError(
            "Não foi possível extrair texto dos documentos."
        )

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    return len(documents)


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Indexa PDFs, TXTs e CSVs do PI-5 no Chroma local."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Apaga o índice local antes de recriá-lo.",
    )

    args = parser.parse_args()

    total = index_documents(
        reset=args.reset
    )

    print(
        f"Índice criado/atualizado com "
        f"{total} chunks em {CHROMA_DIR}"
    )

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

from src.settings import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    INDEX_BATCH_SIZE,
    INDEX_VERSION,
    RAW_DIR,
)


LEGAL_BOUNDARY = re.compile(
    r"(?=^\s*(?:"
    r"Art\.\s*\d+[ºo°]?"
    r"|§\s*\d+[ºo°]?"
    r"|Parágrafo\s+(?:único|\d+)"
    r"|[IVXLCDM]+\s*[-–]"
    r"|[a-z]\)\s*"
    r"))",
    re.IGNORECASE | re.MULTILINE,
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")
READY_FILE = CHROMA_DIR / ".pi5_index_ready.json"
SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".csv",
}


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


# =========================================================
# PDF
# =========================================================

def extract_pdf(
    path: Path,
) -> Iterable[tuple[str, int]]:

    try:
        # Alguns arquivos baixados da web chegam com extensão .pdf,
        # mas na verdade são HTML/erro de download.
        with path.open("rb") as file:
            prefix = file.read(1024)

        if b"%PDF-" not in prefix:
            print(
                f"[AVISO] Arquivo ignorado porque não é PDF válido: "
                f"{path.name}"
            )
            return

        reader = PdfReader(
            str(path),
            strict=False,
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = clean_text(
                    page.extract_text() or ""
                )

                if text:
                    yield text, page_number

            except Exception as error:
                print(
                    f"[AVISO] Erro ao extrair {path.name}, "
                    f"página {page_number}: {error}"
                )

    except Exception as error:
        print(
            f"[AVISO] PDF ignorado ({path.name}): {error}"
        )


# =========================================================
# CSV
# =========================================================

def detect_csv_encoding(path: Path) -> str:
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return "latin-1"


def normalize_column_name(value: str) -> str:
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def find_csv_header(
    rows: list[list[str]],
) -> int:

    for index, row in enumerate(rows):
        normalized = [
            normalize_column_name(cell).lower()
            for cell in row
        ]

        if (
            "uf" in normalized
            and "ente federado" in normalized
        ):
            return index

    for index, row in enumerate(rows):
        filled = [
            cell
            for cell in row
            if cell.strip()
        ]

        if len(filled) >= 2:
            return index

    return 0


def extract_csv(
    path: Path,
) -> Iterable[tuple[str, dict]]:

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
        if not any(
            cell.strip()
            for cell in row
        ):
            continue

        if len(row) < len(headers):
            row += [""] * (
                len(headers) - len(row)
            )

        data: dict[str, str] = {}

        for header, value in zip(
            headers,
            row,
        ):
            header = clean_text(header)
            value = clean_text(value)

            if header and value:
                data[header] = value

        if not data:
            continue

        uf = data.get("UF", "")
        ente = data.get("Ente Federado", "")

        codigo_ibge = (
            data.get("Código IBGE")
            or data.get("Codigo IBGE")
            or data.get("CódigoIBGE")
            or data.get("CodigoIBGE")
            or ""
        )

        parts = [
            "Dados oficiais do VAAR 2026.",
            "Complementação da União ao Fundeb - VAAR.",
        ]

        if ente:
            parts.append(
                f"Ente Federado: {ente}."
            )

        if uf:
            parts.append(
                f"UF: {uf}."
            )

        if codigo_ibge:
            parts.append(
                f"Código IBGE: {codigo_ibge}."
            )

        for column, value in data.items():
            if column in {
                "UF",
                "Ente Federado",
                "Código IBGE",
                "Codigo IBGE",
                "CódigoIBGE",
                "CodigoIBGE",
            }:
                continue

            parts.append(
                f"{column}: {value}."
            )

        text = "\n".join(parts)

        metadata: dict = {
            "source": path.name,
            "type": "csv",
            "row": row_number,
            "page": 1,
            "chunk": 0,
        }

        if uf:
            metadata["uf"] = uf

        if ente:
            metadata["entity"] = ente

        if codigo_ibge:
            metadata["codigo_ibge"] = str(
                codigo_ibge
            )

        yield text, metadata


# =========================================================
# CHUNKING
# =========================================================

def _split_large_unit(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    sentences = SENTENCE_BOUNDARY.split(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if (
            not current
            or len(current) + len(sentence) + 1
            <= chunk_size
        ):
            current = (
                f"{current} {sentence}"
            ).strip()
            continue

        chunks.append(current)

        if overlap > 0:
            prefix = current[-overlap:]
            current = (
                f"{prefix} {sentence}"
            ).strip()
        else:
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def split_legal_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    text = clean_text(text)

    if not text:
        return []

    units = [
        clean_text(unit)
        for unit in LEGAL_BOUNDARY.split(text)
        if clean_text(unit)
    ]

    if not units:
        return _split_large_unit(
            text,
            chunk_size,
            overlap,
        )

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
            continue

        if not current:
            current = unit
            continue

        candidate = (
            current + "\n\n" + unit
        )

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)

            if overlap > 0:
                tail = current[-overlap:]
                current = clean_text(
                    tail + "\n\n" + unit
                )
            else:
                current = unit

    if current:
        chunks.append(current)

    return [
        chunk
        for chunk in chunks
        if len(chunk.strip()) >= 40
    ]


# =========================================================
# DOCUMENTOS
# =========================================================

def document_hint(path: Path) -> str:
    name = path.stem.lower()

    hints: list[str] = []

    if "14.113" in name or "14113" in name:
        hints.append(
            "Documento legal sobre Fundeb, "
            "VAAF, VAAT e VAAR."
        )

    if "imers" in name:
        hints.append(
            "Documento sobre IMERS e ICMS "
            "Educacional do Rio Grande do Sul."
        )

    if "saers" in name:
        hints.append(
            "Documento sobre SAERS e avaliação "
            "educacional do Rio Grande do Sul."
        )

    if (
        "cif" in name
        and "24" in name
    ):
        hints.append(
            "Resolução CIF nº 24/2026 sobre "
            "condicionalidades do VAAR."
        )

    if (
        "cif" in name
        and "25" in name
    ):
        hints.append(
            "Resolução CIF nº 25/2026 sobre "
            "VAAR 2027, atendimento e aprendizagem."
        )

    if "108" in name:
        hints.append(
            "Emenda Constitucional nº 108 "
            "relacionada ao Fundeb e ICMS Educação."
        )

    return " ".join(hints)


def document_records(
    raw_dir: Path,
) -> Iterable[tuple[str, dict]]:

    for path in sorted(
        raw_dir.iterdir()
    ):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        hint = document_hint(path)

        if suffix == ".pdf":
            for page_text, page in extract_pdf(path):
                for index, chunk in enumerate(
                    split_legal_text(page_text)
                ):
                    text = chunk

                    if hint:
                        text = (
                            f"{hint}\n\n{chunk}"
                        )

                    yield text, {
                        "source": path.name,
                        "type": "pdf",
                        "page": page,
                        "chunk": index,
                    }

        elif suffix in {".txt", ".md"}:
            text = clean_text(
                read_text_file(path)
            )

            for index, chunk in enumerate(
                split_legal_text(text)
            ):
                final_text = chunk

                if hint:
                    final_text = (
                        f"{hint}\n\n{chunk}"
                    )

                yield final_text, {
                    "source": path.name,
                    "type": suffix.removeprefix("."),
                    "page": 1,
                    "chunk": index,
                }

        elif suffix == ".csv":
            for text, metadata in extract_csv(path):
                if len(text) >= 20:
                    yield text, metadata


# =========================================================
# ÍNDICE / MANIFESTO
# =========================================================

def raw_manifest_hash(
    raw_dir: Path,
) -> str:

    digest = hashlib.sha256()

    for path in sorted(
        raw_dir.iterdir()
    ):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in SUPPORTED_DOCUMENT_SUFFIXES
        ):
            continue

        digest.update(
            path.name.encode("utf-8")
        )
        digest.update(
            str(path.stat().st_size).encode("utf-8")
        )

        # Hash do conteúdo garante reindexação quando um arquivo muda,
        # mesmo mantendo o mesmo nome.
        file_digest = hashlib.sha256()

        with path.open("rb") as file:
            while True:
                block = file.read(
                    1024 * 1024
                )

                if not block:
                    break

                file_digest.update(block)

        digest.update(
            file_digest.digest()
        )

    return digest.hexdigest()


def read_ready_marker() -> dict | None:
    if not READY_FILE.exists():
        return None

    try:
        return json.loads(
            READY_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


def write_ready_marker(
    *,
    manifest: str,
    count: int,
) -> None:

    payload = {
        "index_version": INDEX_VERSION,
        "manifest": manifest,
        "count": count,
    }

    READY_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_embedding_function(
    max_retries: int = 3,
) -> SentenceTransformerEmbeddingFunction:

    last_error: Exception | None = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            return SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )

        except Exception as error:
            last_error = error

            if attempt < max_retries:
                print(
                    "[AVISO] Falha ao carregar embeddings. "
                    f"Tentativa {attempt}/{max_retries}."
                )
                time.sleep(5)

    raise RuntimeError(
        "Falha ao carregar o modelo de embeddings "
        f"após {max_retries} tentativas: {last_error}"
    )


def create_chroma_client():
    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        return chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

    except Exception as error:
        # Recupera automaticamente bancos SQLite quebrados
        # (ex.: "no such table: tenants").
        message = str(error).lower()

        if (
            "tenants" in message
            or "database" in message
            or "sqlite" in message
        ):
            shutil.rmtree(
                CHROMA_DIR,
                ignore_errors=True,
            )
            CHROMA_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            return chromadb.PersistentClient(
                path=str(CHROMA_DIR)
            )

        raise


def index_documents(
    raw_dir: Path = RAW_DIR,
    reset: bool = False,
    batch_size: int = INDEX_BATCH_SIZE,
) -> int:

    if (
        not raw_dir.exists()
        or not any(raw_dir.iterdir())
    ):
        raise FileNotFoundError(
            f"Nenhum documento encontrado em: "
            f"{raw_dir}"
        )

    manifest = raw_manifest_hash(
        raw_dir
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

    client = create_chroma_client()
    embedding = create_embedding_function()

    marker = read_ready_marker()

    if (
        not reset
        and marker
        and marker.get("index_version")
        == INDEX_VERSION
        and marker.get("manifest")
        == manifest
    ):
        try:
            collection = client.get_collection(
                COLLECTION_NAME,
                embedding_function=embedding,
            )

            count = collection.count()

            if (
                count > 0
                and count
                == int(marker.get("count", 0))
            ):
                print(
                    f"[OK] Índice já pronto com "
                    f"{count} chunks."
                )
                return count

        except Exception:
            pass

    # Se não houver marcador válido, pode ter sobrado uma
    # collection parcialmente indexada de execução interrompida.
    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding,
        metadata={
            "index_version": INDEX_VERSION
        },
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    total = 0

    def flush_batch() -> None:
        nonlocal total

        if not documents:
            return

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        total += len(documents)

        print(
            f"[INDEXAÇÃO] {total} chunks indexados..."
        )

        documents.clear()
        metadatas.clear()
        ids.clear()

    for text, metadata in document_records(
        raw_dir
    ):
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

        if len(documents) >= batch_size:
            flush_batch()

    flush_batch()

    if total == 0:
        raise ValueError(
            "Não foi possível extrair texto "
            "dos documentos."
        )

    write_ready_marker(
        manifest=manifest,
        count=total,
    )

    print(
        f"[OK] Indexação concluída: "
        f"{total} chunks em {CHROMA_DIR}"
    )

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Indexa PDFs, TXTs, MDs e CSVs do PI-5 "
            "no Chroma."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Apaga o índice antes de "
            "recriá-lo."
        ),
    )

    args = parser.parse_args()

    total = index_documents(
        reset=args.reset
    )

    print(
        f"Índice criado/atualizado com "
        f"{total} chunks."
    )

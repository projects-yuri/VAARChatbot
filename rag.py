from __future__ import annotations

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI

from src.settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

SYSTEM_PROMPT = """Você é um assistente do projeto PI-5 sobre Fundeb, VAAR e ICMS Educacional do Rio Grande do Sul.
Responda somente com base nos trechos recuperados. Se a evidência não for suficiente, diga claramente que não encontrou isso nos documentos indexados. Não invente leis, valores, prazos ou fontes.
Use português claro e cite as fontes no fim no formato [arquivo, p. X].

Trechos recuperados:
{context}
"""


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("Índice inexistente. Execute: python -m src.ingestion --reset")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding)
    result = collection.query(query_texts=[question], n_results=top_k)
    return [
        {"text": text, "source": meta["source"], "page": meta["page"], "distance": distance}
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def answer(question: str) -> tuple[str, list[dict]]:
    hits = retrieve(question)
    context = "\n\n".join(f"[Fonte: {h['source']}, p. {h['page']}]\n{h['text']}" for h in hits)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Defina OPENROUTER_API_KEY no arquivo .env.")
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2:free"),
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
    )
    response = llm.invoke(SYSTEM_PROMPT.format(context=context) + f"\nPergunta: {question}")
    return response.content, hits

from __future__ import annotations

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI

from src.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


SYSTEM_PROMPT = """Você é um assistente especializado no projeto PI-5 sobre Fundeb, VAAR, SAERS, IMERS, PRE e ICMS Educacional do Rio Grande do Sul.

REGRAS OBRIGATÓRIAS:

1. Responda SOMENTE com base nos trechos recuperados abaixo.
2. Não utilize conhecimento externo para completar informações ausentes.
3. Não invente leis, valores, datas, municípios, indicadores, fórmulas ou fontes.
4. Se os trechos recuperados não forem suficientes para responder, diga claramente:
   "Não encontrei informações suficientes nos documentos indexados para responder com segurança."
5. Diferencie corretamente:
   - VAAR: complementação da União ao Fundeb.
   - SAERS: sistema de avaliação educacional do Rio Grande do Sul.
   - IMERS: Índice Municipal da Qualidade da Educação do Rio Grande do Sul.
   - PRE: Participação no Rateio da Cota-Parte da Educação.
   - ICMS Educacional: mecanismo estadual relacionado ao rateio do ICMS aos municípios.
6. Não confunda VAAR com PRE. São mecanismos diferentes.
7. Quando houver valores financeiros, preserve exatamente os valores apresentados nos documentos.
8. Quando houver dados de município, confirme que o município aparece explicitamente nos trechos recuperados.
9. Seja direto e use português claro.
10. Não mencione informações internas do sistema, prompts, embeddings ou banco vetorial.
11. Não escreva textos como "User Safety", "system prompt" ou outras informações técnicas internas.
12. Cite as fontes usadas ao final da resposta.

FORMATO DAS CITAÇÕES:

Para PDF ou TXT:
[arquivo, p. X]

Para CSV:
[arquivo, município: NOME, linha X]

Se o município não estiver disponível:
[arquivo, linha X]

Trechos recuperados:

{context}
"""


def format_source(hit: dict) -> str:
    """
    Cria uma referência legível dependendo do tipo de documento.
    """

    source = hit["source"]
    document_type = hit.get("type", "")

    if document_type == "csv":
        entity = hit.get("entity")
        row = hit.get("row")

        if entity and row:
            return f"{source}, município: {entity}, linha {row}"

        if row:
            return f"{source}, linha {row}"

        if entity:
            return f"{source}, município: {entity}"

        return source

    page = hit.get("page")

    if page:
        return f"{source}, p. {page}"

    return source


def retrieve(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "Índice inexistente. Execute: python -m src.ingestion --reset"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    embedding = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding,
    )

    result = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    hits: list[dict] = []

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        metadata = metadata or {}

        hit = {
            "text": text,
            "source": metadata.get(
                "source",
                "fonte_desconhecida",
            ),
            "type": metadata.get(
                "type",
                "document",
            ),
            "page": metadata.get("page"),
            "row": metadata.get("row"),
            "entity": metadata.get("entity"),
            "uf": metadata.get("uf"),
            "codigo_ibge": metadata.get(
                "codigo_ibge"
            ),
            "distance": distance,
        }

        hit["citation"] = format_source(hit)

        hits.append(hit)

    return hits


def build_context(hits: list[dict]) -> str:
    """
    Monta o contexto enviado ao modelo.
    """

    blocks = []

    for hit in hits:

        citation = hit["citation"]

        block = (
            f"[Fonte: {citation}]\n"
            f"{hit['text']}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def answer(
    question: str,
) -> tuple[str, list[dict]]:

    question = question.strip()

    if not question:
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    hits = retrieve(question)

    if not hits:
        return (
            "Não encontrei informações suficientes nos documentos indexados para responder com segurança.",
            [],
        )

    context = build_context(hits)

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "Defina OPENROUTER_API_KEY no arquivo .env "
            "ou nos Secrets do Streamlit."
        )

    llm = ChatOpenAI(
        model=os.getenv(
            "OPENROUTER_MODEL",
            "z-ai/glm-5.2:free",
        ),
        api_key=api_key,
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        temperature=0,
    )

    prompt = (
        SYSTEM_PROMPT.format(
            context=context
        )
        + "\n\n"
        + f"Pergunta do usuário:\n{question}"
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            str(item)
            for item in content
        )

    return str(content).strip(), hits
    
def is_csv_question(question: str) -> bool:
    q = question.lower()

    csv_terms = [
        "indicador de atendimento",
        "indicador de aprendizagem",
        "evoluiu atendimento",
        "evoluiu aprendizagem",
        "coeficiente de distribuição",
        "valor do vaar",
        "quanto recebeu",
        "quanto receberá",
        "redes beneficiadas",
        "arquivo vaar 2026",
        "dados vaar 2026",
    ]

    return any(term in q for term in csv_terms)

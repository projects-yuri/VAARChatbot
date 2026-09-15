from __future__ import annotations

import os
import re
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI

from src.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


SYSTEM_PROMPT = """
Você é um assistente especializado na documentação do projeto PI-5 sobre
Fundeb, VAAR, SAERS, IMERS, PRE e ICMS Educacional do Rio Grande do Sul.

REGRAS OBRIGATÓRIAS:
1. Responda somente com base nos trechos recuperados.
2. Não invente dados, percentuais, datas, municípios, fórmulas ou conceitos.
3. Se os trechos não forem suficientes, diga claramente que não encontrou
   informação suficiente na documentação recuperada.
4. Para perguntas conceituais, legais ou metodológicas, priorize legislação,
   decretos, resoluções e documentos textuais.
5. Use dados CSV somente quando eles estiverem presentes nos trechos
   recuperados e a pergunta for sobre valores, indicadores, municípios,
   coeficientes ou redes beneficiadas.
6. Não confunda VAAR com PRE:
   - VAAR é uma modalidade de complementação da União ao Fundeb.
   - PRE pertence ao contexto do ICMS Educacional do Rio Grande do Sul.
7. Não trate exemplos de municípios como definição geral de um conceito.
8. Ao citar uma informação, use apenas as referências fornecidas no contexto.
9. Seja direto e organizado. Não acrescente exemplos que não foram pedidos.
"""


@lru_cache(maxsize=1)
def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_collection():
    """
    Abre a coleção do Chroma.

    Se a coleção ainda não existir (por exemplo, após trocar o CHROMA_DIR
    ou reiniciar o Streamlit Cloud), dispara a indexação uma vez e tenta
    abrir novamente.
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        return client.get_collection(
            COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )

    except Exception as first_error:
        # Importação local evita dependência circular na carga do módulo.
        from src.ingestion import index_documents

        index_documents(reset=False)

        # Reabre o cliente depois da indexação.
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        try:
            return client.get_collection(
                COLLECTION_NAME,
                embedding_function=get_embedding_function(),
            )

        except Exception as second_error:
            raise RuntimeError(
                "Não foi possível criar ou abrir a coleção "
                f"'{COLLECTION_NAME}' no Chroma. "
                f"Erro inicial: {first_error}. "
                f"Erro após indexação: {second_error}"
            ) from second_error


def is_csv_question(question: str) -> bool:
    q = question.lower()

    csv_terms = [
        "indicador de atendimento",
        "indicador de aprendizagem",
        "evoluiu atendimento",
        "evoluiu aprendizagem",
        "evoluiuatendiment",
        "evoluiuaprendizagem",
        "coeficiente de distribuição",
        "coeficiente do vaar",
        "valor do vaar",
        "valor de vaar",
        "quanto recebeu",
        "quanto recebe",
        "quanto receberá",
        "rede beneficiada",
        "redes beneficiadas",
        "municípios beneficiados",
        "municipios beneficiados",
        "código ibge",
        "codigo ibge",
        "dados vaar 2026",
        "arquivo vaar 2026",
    ]

    return any(term in q for term in csv_terms)


def build_queries(question: str) -> list[str]:
    q = question.lower()
    queries = [question]
    expansions: list[str] = []

    if "fundeb" in q:
        expansions.append(
            "Fundeb Lei 14.113 de 2020 finalidade complementação da União "
            "VAAF VAAT VAAR"
        )

    if "vaar" in q:
        expansions.append(
            "complementação VAAR Lei 14.113 de 2020 art. 5 art. 14 "
            "condicionalidades melhoria de gestão atendimento aprendizagem "
            "redução das desigualdades"
        )

    if "saers" in q:
        expansions.append(
            "SAERS Sistema de Avaliação do Rendimento Escolar do "
            "Rio Grande do Sul finalidade avaliação"
        )

    if "imers" in q:
        expansions.append(
            "IMERS Índice Municipal da Qualidade da Educação do "
            "Rio Grande do Sul cálculo finalidade"
        )

    if re.search(r"\bpre\b", q):
        expansions.append(
            "PRE Participação no Rateio da Cota-Parte da Educação "
            "ICMS Educacional Rio Grande do Sul"
        )

    if "icms" in q:
        expansions.append(
            "ICMS Educacional Rio Grande do Sul IMERS PRE "
            "resultados de aprendizagem equidade distribuição"
        )

    if "resolução cif nº 24" in q or "resolucao cif 24" in q or "cif 24" in q:
        expansions.append(
            "Resolução CIF nº 24 de 2026 VAAR condicionalidades "
            "gestores escolares Simec"
        )

    if "resolução cif nº 25" in q or "resolucao cif 25" in q or "cif 25" in q:
        expansions.append(
            "Resolução CIF nº 25 de 2026 VAAR 2027 "
            "condicionalidades II III atendimento aprendizagem Saeb"
        )

    for item in expansions:
        if item not in queries:
            queries.append(item)

    return queries


def format_source(metadata: dict) -> str:
    source = metadata.get("source", "Documento")
    doc_type = metadata.get("type", "document")

    if doc_type == "csv":
        entity = metadata.get("entity")
        row = metadata.get("row")
        details = []

        if entity:
            details.append(f"município: {entity}")
        if row:
            details.append(f"linha {row}")

        if details:
            return f"[{source}, {', '.join(details)}]"
        return f"[{source}]"

    if doc_type == "pdf":
        page = metadata.get("page")
        if page:
            return f"[{source}, p. {page}]"

    return f"[{source}]"


def _hit_key(metadata: dict) -> tuple:
    return (
        metadata.get("source"),
        metadata.get("type"),
        metadata.get("page"),
        metadata.get("row"),
        metadata.get("chunk"),
    )


def retrieve(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:

    collection = get_collection()
    csv_question = is_csv_question(question)

    if csv_question:
        document_filter = {"type": "csv"}
        queries = [question]
        final_limit = min(max(top_k, 5), 8)
        per_query = final_limit
    else:
        document_filter = {
            "type": {
                "$in": ["pdf", "txt"]
            }
        }
        queries = build_queries(question)
        final_limit = max(top_k, 8)
        final_limit = min(final_limit, 12)
        per_query = 5

    grouped_hits: list[list[dict]] = []

    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=per_query,
            where=document_filter,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        current_hits: list[dict] = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            current_hits.append({
                "text": text,
                "source": metadata.get("source", "Documento"),
                "type": metadata.get("type", "document"),
                "page": metadata.get("page"),
                "row": metadata.get("row"),
                "chunk": metadata.get("chunk"),
                "entity": metadata.get("entity"),
                "uf": metadata.get("uf"),
                "codigo_ibge": metadata.get("codigo_ibge"),
                "distance": distance,
                "citation": format_source(metadata),
                "_metadata": metadata,
            })

        grouped_hits.append(current_hits)

    selected: list[dict] = []
    seen: set[tuple] = set()

    if not csv_question and len(grouped_hits) > 1:
        for group in grouped_hits:
            for hit in group[:2]:
                key = _hit_key(hit["_metadata"])

                if key not in seen:
                    selected.append(hit)
                    seen.add(key)

                if len(selected) >= final_limit:
                    break

            if len(selected) >= final_limit:
                break

    all_hits = [
        hit
        for group in grouped_hits
        for hit in group
    ]

    all_hits.sort(
        key=lambda item: (
            item["distance"]
            if item["distance"] is not None
            else float("inf")
        )
    )

    for hit in all_hits:
        if len(selected) >= final_limit:
            break

        key = _hit_key(hit["_metadata"])

        if key in seen:
            continue

        selected.append(hit)
        seen.add(key)

    for hit in selected:
        hit.pop("_metadata", None)

    return selected


def build_context(hits: list[dict]) -> str:
    blocks = []

    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"TRECHO {index}\n"
            f"Fonte: {hit['citation']}\n"
            f"{hit['text']}"
        )

    return "\n\n".join(blocks)


def answer(question: str):
    hits = retrieve(question)

    if not hits:
        return (
            "Não encontrei informações suficientes nos documentos "
            "indexados para responder com segurança.",
            [],
        )

    context = build_context(hits)
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não configurada."
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

    user_prompt = f"""
PERGUNTA:
{question}

TRECHOS RECUPERADOS:
{context}

INSTRUÇÕES PARA A RESPOSTA:
- Responda diretamente à pergunta.
- Use somente os trechos recuperados acima.
- Não transforme exemplos municipais em regras gerais.
- Se houver legislação ou resolução nos trechos, use-a como fonte principal
  para conceitos, percentuais, requisitos e metodologias.
- Não acrescente dados de municípios se a pergunta não pedir.
- Ao final de cada afirmação importante, cite a fonte correspondente usando
  exatamente o formato de referência fornecido em "Fonte:".
- Se os trechos forem insuficientes, diga isso claramente.
"""

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    text = response.content

    if not isinstance(text, str):
        text = str(text)

    return text.strip(), hits

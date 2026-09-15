from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI

from src.settings import (
    CANDIDATE_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHUNKS,
    TOP_K,
)


SYSTEM_PROMPT = """
Você é um assistente especializado na documentação do projeto PI-5
sobre Fundeb, VAAR, SAERS, IMERS, PRE e ICMS Educacional do
Rio Grande do Sul.

REGRAS OBRIGATÓRIAS:
1. Responda somente com base nos trechos recuperados.
2. Não invente dados, percentuais, datas, municípios, fórmulas ou conceitos.
3. Se os trechos forem insuficientes, diga claramente que a documentação
   recuperada não foi suficiente.
4. Para perguntas conceituais, legais ou metodológicas, priorize legislação,
   decretos, resoluções e documentos textuais.
5. CSV deve ser usado para perguntas sobre valores, indicadores, municípios,
   coeficientes ou redes beneficiadas.
6. Não confunda VAAR com PRE:
   - VAAR é modalidade de complementação da União ao Fundeb.
   - PRE pertence ao contexto do ICMS Educacional do Rio Grande do Sul.
7. Não transforme exemplo de município em regra geral.
8. Não acrescente exemplos que a pergunta não solicitou.
9. Cite somente as fontes presentes no contexto fornecido.
10. Quando a pergunta exigir relacionar documentos, combine os trechos
    recuperados em uma explicação única e coerente.
"""


# =========================================================
# CHROMA
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_collection():
    """
    Garante que a collection exista antes de fazer a busca.
    Se o Streamlit reiniciar e /tmp estiver vazio, indexa automaticamente.
    """
    from src.ingestion import index_documents

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        return client.get_collection(
            COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )

    except Exception:
        index_documents(
            reset=False
        )

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        return client.get_collection(
            COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )


def clear_rag_caches() -> None:
    get_collection.cache_clear()
    get_embedding_function.cache_clear()


# =========================================================
# ROTEAMENTO
# =========================================================

def is_csv_question(
    question: str,
) -> bool:

    q = question.lower()

    csv_terms = [
        "indicador de atendimento",
        "indicador atendimento",
        "indicador de aprendizagem",
        "indicador aprendizagem",
        "evoluiu atendimento",
        "evoluiu aprendizagem",
        "coeficiente de distribuição",
        "coeficiente do vaar",
        "valor do vaar",
        "quanto recebeu",
        "quanto recebe",
        "quanto receberá",
        "redes beneficiadas",
        "rede beneficiada",
        "municípios beneficiados",
        "municipios beneficiados",
        "código ibge",
        "codigo ibge",
        "dados vaar 2026",
        "arquivo vaar 2026",
    ]

    return any(
        term in q
        for term in csv_terms
    )


def build_queries(
    question: str,
) -> list[str]:

    q = question.lower()
    queries = [question]

    expansions: list[str] = []

    if "fundeb" in q:
        expansions.append(
            "Fundeb Lei 14.113 2020 finalidade "
            "VAAF VAAT VAAR complementação da União"
        )

    if "vaar" in q:
        expansions.append(
            "VAAR Lei 14.113 art. 5 art. 14 "
            "2,5 pontos percentuais condicionalidades "
            "gestão atendimento aprendizagem desigualdades"
        )

    if "saers" in q:
        expansions.append(
            "SAERS Sistema de Avaliação do Rendimento "
            "Escolar do Rio Grande do Sul finalidade "
            "avaliação aprendizagem"
        )

    if "imers" in q:
        expansions.append(
            "IMERS Índice Municipal da Qualidade da Educação "
            "do Rio Grande do Sul finalidade cálculo "
            "proficiência aprovação evolução"
        )

    if re.search(
        r"\bpre\b",
        q,
    ):
        expansions.append(
            "PRE Participação no Rateio da Cota-Parte "
            "da Educação ICMS Educacional Rio Grande do Sul"
        )

    if "icms" in q:
        expansions.append(
            "ICMS Educacional Rio Grande do Sul "
            "IMERS PRE aprendizagem equidade distribuição"
        )

    if (
        "cif" in q
        and "24" in q
    ):
        expansions.append(
            "Resolução CIF 24 2026 "
            "condicionalidades I IV V VAAR "
            "gestores escolares Simec"
        )

    if (
        "cif" in q
        and "25" in q
    ):
        expansions.append(
            "Resolução CIF 25 2026 "
            "condicionalidades II III VAAR 2027 "
            "Saeb atendimento aprendizagem"
        )

    for expansion in expansions:
        if expansion not in queries:
            queries.append(expansion)

    return queries


# =========================================================
# CITAÇÃO / RERANKING
# =========================================================

def format_source(
    metadata: dict,
) -> str:

    source = metadata.get(
        "source",
        "Documento",
    )

    doc_type = metadata.get(
        "type",
        "document",
    )

    if doc_type == "csv":
        details: list[str] = []

        entity = metadata.get("entity")
        row = metadata.get("row")

        if entity:
            details.append(
                f"município: {entity}"
            )

        if row:
            details.append(
                f"linha {row}"
            )

        if details:
            return (
                f"[{source}, "
                f"{', '.join(details)}]"
            )

        return f"[{source}]"

    if doc_type == "pdf":
        page = metadata.get("page")

        if page:
            return (
                f"[{source}, p. {page}]"
            )

    return f"[{source}]"


def hit_key(
    hit: dict,
) -> tuple:

    return (
        hit.get("source"),
        hit.get("type"),
        hit.get("page"),
        hit.get("row"),
        hit.get("chunk"),
    )


STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como",
    "da", "das", "de", "do", "dos", "e",
    "em", "é", "na", "nas", "no", "nos",
    "o", "os", "para", "por", "qual",
    "quais", "que", "sua", "seu", "um",
    "uma", "no", "rio", "grande", "sul",
}


def normalize_terms(
    text: str,
) -> set[str]:

    text = text.lower()

    text = (
        text.replace("º", "")
        .replace("ª", "")
    )

    terms = set(
        re.findall(
            r"[a-zà-ÿ0-9]+",
            text,
        )
    )

    return {
        term
        for term in terms
        if (
            term not in STOPWORDS
            and len(term) >= 3
        )
    }


def lexical_score(
    question: str,
    text: str,
) -> float:

    q_terms = normalize_terms(
        question
    )

    if not q_terms:
        return 0.0

    text_terms = normalize_terms(
        text
    )

    overlap = len(
        q_terms & text_terms
    )

    return overlap / len(q_terms)


def source_boost(
    question: str,
    source: str,
) -> float:

    q = question.lower()
    s = source.lower()

    boost = 0.0

    pairs = [
        ("imers", "imers"),
        ("saers", "saers"),
        ("vaar", "14113"),
        ("vaar", "14.113"),
        ("fundeb", "14113"),
        ("fundeb", "14.113"),
        ("cif", "cif"),
        ("24", "24"),
        ("25", "25"),
        ("icms", "imers"),
        ("icms", "saers"),
    ]

    for q_term, s_term in pairs:
        if (
            q_term in q
            and s_term in s
        ):
            boost += 0.12

    return min(
        boost,
        0.35,
    )


def ranking_score(
    question: str,
    hit: dict,
) -> float:

    distance = hit.get("distance")

    if distance is None:
        distance = 99.0

    lexical = lexical_score(
        question,
        (
            f"{hit.get('source', '')} "
            f"{hit.get('text', '')}"
        ),
    )

    boost = source_boost(
        question,
        hit.get("source", ""),
    )

    # Menor distância é melhor; maior score léxico é melhor.
    return (
        -float(distance)
        + (lexical * 0.70)
        + boost
    )


# =========================================================
# RECUPERAÇÃO
# =========================================================

def retrieve(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:

    collection = get_collection()

    csv_question = is_csv_question(
        question
    )

    if csv_question:
        document_filter = {
            "type": "csv"
        }
        queries = [question]
        final_limit = min(
            max(top_k, 8),
            MAX_CONTEXT_CHUNKS,
        )
        per_query = min(
            max(CANDIDATE_K, final_limit),
            35,
        )

    else:
        document_filter = {
            "type": {
                "$in": [
                    "pdf",
                    "txt",
                ]
            }
        }

        queries = build_queries(
            question
        )

        final_limit = min(
            max(top_k, 12),
            MAX_CONTEXT_CHUNKS,
        )

        per_query = min(
            max(CANDIDATE_K, 20),
            35,
        )

    candidates: list[dict] = []

    for query in queries:
        try:
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
        except Exception:
            # Em coleções pequenas, algumas versões do Chroma
            # podem reclamar de n_results alto.
            results = collection.query(
                query_texts=[query],
                n_results=max(
                    final_limit,
                    5,
                ),
                where=document_filter,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

        documents = (
            results.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            results.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            results.get(
                "distances",
                [[]],
            )[0]
        )

        for (
            text,
            metadata,
            distance,
        ) in zip(
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            hit = {
                "text": text,
                "source": metadata.get(
                    "source",
                    "Documento",
                ),
                "type": metadata.get(
                    "type",
                    "document",
                ),
                "page": metadata.get("page"),
                "row": metadata.get("row"),
                "chunk": metadata.get("chunk"),
                "entity": metadata.get("entity"),
                "uf": metadata.get("uf"),
                "codigo_ibge": metadata.get(
                    "codigo_ibge"
                ),
                "distance": distance,
            }

            hit["citation"] = (
                format_source(metadata)
            )

            hit["score"] = (
                ranking_score(
                    question,
                    hit,
                )
            )

            candidates.append(hit)

    # Deduplicação.
    unique: dict[tuple, dict] = {}

    for hit in candidates:
        key = hit_key(hit)

        if (
            key not in unique
            or hit["score"]
            > unique[key]["score"]
        ):
            unique[key] = hit

    ranked = sorted(
        unique.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    # Diversidade de fontes: evita que 12 chunks do mesmo documento
    # dominem uma pergunta que exige cruzamento documental.
    selected: list[dict] = []
    per_source: defaultdict[str, int] = defaultdict(int)

    max_per_source = (
        6
        if len(build_queries(question)) == 1
        else 4
    )

    for hit in ranked:
        source = hit["source"]

        if (
            per_source[source]
            >= max_per_source
        ):
            continue

        selected.append(hit)
        per_source[source] += 1

        if len(selected) >= final_limit:
            break

    # Se a diversidade deixou poucos trechos, completa com os melhores.
    if len(selected) < final_limit:
        selected_keys = {
            hit_key(hit)
            for hit in selected
        }

        for hit in ranked:
            key = hit_key(hit)

            if key in selected_keys:
                continue

            selected.append(hit)
            selected_keys.add(key)

            if len(selected) >= final_limit:
                break

    for hit in selected:
        hit.pop("score", None)

    return selected


# =========================================================
# CONTEXTO / LLM
# =========================================================

def build_context(
    hits: list[dict],
) -> str:

    blocks: list[str] = []

    for index, hit in enumerate(
        hits,
        start=1,
    ):
        blocks.append(
            f"TRECHO {index}\n"
            f"Fonte: {hit['citation']}\n"
            f"{hit['text']}"
        )

    return "\n\n".join(blocks)


def answer(
    question: str,
):

    hits = retrieve(
        question
    )

    if not hits:
        return (
            "Não encontrei informações suficientes "
            "nos documentos indexados para responder "
            "com segurança.",
            [],
        )

    context = build_context(
        hits
    )

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

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

INSTRUÇÕES:
- Responda diretamente à pergunta.
- Use somente os trechos recuperados.
- Considere todos os trechos relevantes, não apenas o primeiro.
- Para conceitos, percentuais, requisitos e metodologias, prefira
  legislação, decretos e resoluções presentes nos trechos.
- Não use exemplos de municípios como definição de conceitos.
- Não acrescente dados municipais se a pergunta não solicitar.
- Se a pergunta pedir comparação ou relação entre temas, combine
  informações de mais de um trecho quando necessário.
- Cite as fontes usando exatamente os rótulos apresentados em "Fonte:".
- Se a documentação recuperada realmente não trouxer a informação,
  diga que ela é insuficiente.
"""

    response = llm.invoke(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                user_prompt,
            ),
        ]
    )

    text = response.content

    if not isinstance(
        text,
        str,
    ):
        text = str(text)

    return (
        text.strip(),
        hits,
    )

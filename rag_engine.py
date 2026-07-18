from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def get_openai_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=key)


def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _format_context(results: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    source_chunks: list[dict[str, Any]] = []
    formatted_parts: list[str] = []

    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        distance = distances[index] if index < len(distances) else None
        source_chunks.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )
        formatted_parts.append(
            f"[Source {index + 1}] Document: {metadata.get('document_name', 'Unknown')}\n"
            f"Equipment Tags: {metadata.get('equipment_tags', '')}\n"
            f"Chunk: {document}"
        )

    return "\n\n".join(formatted_parts), source_chunks


def ask_question(
    question: str,
    chroma_collection: Any,
    client: OpenAI | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    llm_client = client
    if llm_client is None:
        llm_client = get_openai_client()

    question_embedding = embed_text(llm_client, question)
    results = chroma_collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    context_text, source_chunks = _format_context(results)
    if not source_chunks:
        return {
            "answer": "Information not found in current documents.",
            "source_chunks": [],
            "related_equipment_tags": [],
            "source_documents": [],
        }

    system_prompt = (
        "You are a senior industrial engineer. Answer based ONLY on the provided context. "
        "If the answer is not in the context, reply exactly: Information not found in current documents. "
        "Do not invent facts. Cite document names inline in the answer. "
        "Return a concise answer with these sections:\n"
        "1. Direct Answer\n"
        "2. Source Document Name\n"
        "3. Related Equipment Tags"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Write the response using only the context above."
    )

    response = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = "Information not found in current documents."

    source_documents = []
    equipment_tags: set[str] = set()
    for chunk in source_chunks:
        metadata = chunk.get("metadata", {})
        source_documents.append(
            {
                "document_name": metadata.get("document_name", "Unknown"),
                "equipment_tags": metadata.get("equipment_tags", ""),
                "chunk_text": chunk.get("text", ""),
                "distance": chunk.get("distance"),
            }
        )
        tags = metadata.get("equipment_tags", "")
        if isinstance(tags, str) and tags:
            equipment_tags.update(tag for tag in tags.split("|") if tag)

    return {
        "answer": answer,
        "source_chunks": source_chunks,
        "source_documents": source_documents,
        "related_equipment_tags": sorted(equipment_tags),
    }

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

import chromadb


DEFAULT_CHROMA_DIR = ".chroma_db"
DEFAULT_COLLECTION_NAME = "industrial_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
VISION_MODEL = "gpt-4o-mini"

load_dotenv()


@dataclass
class ParsedDocument:
    name: str
    source_type: str
    text: str
    source_hash: str


def get_openai_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=key)


def get_chroma_client(chroma_dir: str = DEFAULT_CHROMA_DIR) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=chroma_dir)


def get_collection(
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> chromadb.Collection:
    client = get_chroma_client(chroma_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _source_hash(name: str, data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(name.encode("utf-8"))
    digest.update(data)
    return digest.hexdigest()


def _normalize_text_rows(rows: Iterable[dict[str, Any]], prefix: str = "") -> str:
    lines: list[str] = []
    for row in rows:
        values = []
        for key, value in row.items():
            if pd.isna(value):
                continue
            values.append(f"{key}: {value}")
        if values:
            lines.append(f"{prefix}{'; '.join(values)}")
    return "\n".join(lines)


def extract_pdf_text(file_path: str | Path, file_bytes: bytes | None = None) -> str:
    if fitz is not None:
        if file_bytes is not None:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        else:
            doc = fitz.open(str(file_path))
        try:
            return "\n".join(page.get_text("text") for page in doc)
        finally:
            doc.close()

    if PdfReader is None:
        raise ImportError("Neither PyMuPDF nor pypdf is available for PDF parsing.")

    if file_bytes is not None:
        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
    else:
        reader = PdfReader(str(file_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_tabular_text(file_path: str | Path, file_bytes: bytes | None = None) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        if file_bytes is None:
            df = pd.read_csv(file_path)
        else:
            from io import BytesIO

            df = pd.read_csv(BytesIO(file_bytes))
        return _normalize_text_rows(df.to_dict(orient="records"))

    if file_bytes is None:
        sheets = pd.read_excel(file_path, sheet_name=None)
    else:
        from io import BytesIO

        sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)

    sheet_texts: list[str] = []
    for sheet_name, df in sheets.items():
        sheet_texts.append(f"Sheet: {sheet_name}")
        sheet_texts.append(_normalize_text_rows(df.to_dict(orient="records"), prefix="Row: "))
    return "\n".join(part for part in sheet_texts if part)


def _image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def describe_image_with_vision(
    client: OpenAI,
    image_bytes: bytes,
    filename: str,
    mime_type: str,
) -> str:
    response = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract industrial document content from P&ID and plant images. "
                    "Return only concise factual text. Do not invent any labels."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all equipment tags, valves, and process flow descriptions "
                            "from this P&ID diagram."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_bytes_to_data_url(image_bytes, mime_type),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content or f"Image: {filename}"


def _heuristic_entities(text: str) -> dict[str, list[str]]:
    tags = sorted(set(re.findall(r"\b[A-Z]{1,3}-\d{2,4}\b", text)))
    keywords = {
        "maintenance": ["maintenance", "repair", "inspection", "overhaul", "alignment", "lubrication"],
        "safety": ["safety", "loto", "lock-out", "tag-out", "ppe", "guard", "hazard"],
        "failure_mode": ["leak", "leakage", "vibration", "overheat", "temperature", "corrosion", "seal"],
        "process": ["pump", "valve", "pressure", "flow", "compressor", "vessel", "motor"],
    }
    types = [label for label, words in keywords.items() if any(word in text.lower() for word in words)]
    failure_modes = ["seal leakage"] if "leak" in text.lower() else []
    hazards = []
    if any(word in text.lower() for word in ["guard", "rotating", "pressure", "hot", "entanglement"]):
        hazards = ["rotating equipment", "pressure release", "hot surfaces"]
    return {
        "tags": tags,
        "type": types,
        "failure_modes": failure_modes,
        "safety_hazards": hazards,
    }


def extract_entities_for_chunk(
    client: OpenAI | None,
    chunk_text_value: str,
    document_name: str,
) -> dict[str, list[str]]:
    if client is None:
        return _heuristic_entities(chunk_text_value)

    prompt = (
        "Extract industrial entities from the text chunk below and return JSON only. "
        'Use this schema: {"tags": [], "type": [], "failure_modes": [], "safety_hazards": []}. '
        "tags must contain equipment tags like P-101 or V-303. type should contain short labels "
        "such as maintenance, safety, operations, inspection, process, or reliability. "
        "Keep arrays short and factual. If nothing is found, return empty arrays.\n\n"
        f"Document: {document_name}\n"
        f"Chunk: {chunk_text_value}"
    )

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an information extraction engine. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        payload = response.choices[0].message.content or "{}"
        parsed = json.loads(payload)
    except Exception:
        return _heuristic_entities(chunk_text_value)

    entities = {
        "tags": [str(item) for item in parsed.get("tags", []) if str(item).strip()],
        "type": [str(item) for item in parsed.get("type", []) if str(item).strip()],
        "failure_modes": [str(item) for item in parsed.get("failure_modes", []) if str(item).strip()],
        "safety_hazards": [str(item) for item in parsed.get("safety_hazards", []) if str(item).strip()],
    }
    return entities


def _metadata_from_entities(
    *,
    document_name: str,
    source_type: str,
    source_hash: str,
    chunk_index: int,
    entities: dict[str, list[str]],
    uploaded_by: str = "",
) -> dict[str, str | int]:
    return {
        "document_name": document_name,
        "source_type": source_type,
        "source_hash": source_hash,
        "chunk_index": chunk_index,
        "uploaded_by": uploaded_by,
        "equipment_tags": "|".join(entities.get("tags", [])),
        "entity_types": "|".join(entities.get("type", [])),
        "failure_modes": "|".join(entities.get("failure_modes", [])),
        "safety_hazards": "|".join(entities.get("safety_hazards", [])),
        "entities_json": json.dumps(entities, ensure_ascii=True),
    }


def _make_chunk_id(source_hash: str, chunk_index: int) -> str:
    return f"{source_hash[:16]}-{chunk_index}"


def _parse_source(
    source: str | Path | Any,
    client: OpenAI | None,
) -> ParsedDocument:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        file_bytes = path.read_bytes()
        name = path.name
        suffix = path.suffix.lower()
    else:
        name = getattr(source, "name", "uploaded_file")
        suffix = Path(name).suffix.lower()
        file_bytes = source.getvalue() if hasattr(source, "getvalue") else source.read()
        path = Path(name)

    source_hash = _source_hash(name, file_bytes)

    if suffix == ".pdf":
        text = extract_pdf_text(path, file_bytes=file_bytes)
    elif suffix in {".csv", ".xls", ".xlsx"}:
        text = extract_tabular_text(path, file_bytes=file_bytes)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        if client is None:
            raise ValueError("OPENAI_API_KEY is required to process image documents.")
        text = describe_image_with_vision(client, file_bytes, name, mime_type=f"image/{suffix.lstrip('.')}")
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    return ParsedDocument(name=name, source_type=suffix.lstrip("."), text=text, source_hash=source_hash)


def ingest_sources(
    sources: Sequence[str | Path | Any],
    *,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: OpenAI | None = None,
    chunk_size: int = 1000,
    overlap: int = 200,
    uploaded_by: str = "",
) -> dict[str, Any]:
    collection = get_collection(chroma_dir=chroma_dir, collection_name=collection_name)
    llm_client = client
    if llm_client is None:
        try:
            llm_client = get_openai_client()
        except ValueError:
            llm_client = None

    added = 0
    processed_documents: list[str] = []
    processed_entries: list[dict[str, str | int]] = []
    warnings: list[str] = []

    for source in sources:
        if source is None:
            continue
        parsed = _parse_source(source, llm_client)
        processed_documents.append(parsed.name)
        chunks = chunk_text(parsed.text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            warnings.append(f"No text extracted from {parsed.name}.")
            processed_entries.append(
                {
                    "document_name": parsed.name,
                    "source_type": parsed.source_type,
                    "chunks_added": 0,
                }
            )
            continue

        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        ids: list[str] = []

        for idx, chunk in enumerate(chunks):
            entities = extract_entities_for_chunk(llm_client, chunk, parsed.name)
            metadata = _metadata_from_entities(
                document_name=parsed.name,
                source_type=parsed.source_type,
                source_hash=parsed.source_hash,
                chunk_index=idx,
                entities=entities,
                uploaded_by=uploaded_by.strip(),
            )
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(_make_chunk_id(parsed.source_hash, idx))

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        added += len(ids)
        processed_entries.append(
            {
                "document_name": parsed.name,
                "source_type": parsed.source_type,
                "chunks_added": len(ids),
            }
        )

    return {
        "processed_documents": processed_documents,
        "processed_entries": processed_entries,
        "chunks_added": added,
        "warnings": warnings,
    }


def ingest_folder(
    folder_path: str | Path,
    *,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    sources = [path for path in folder.iterdir() if path.is_file()]
    return ingest_sources(
        sources,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        client=client,
    )


def get_unique_entities(
    collection: chromadb.Collection,
) -> dict[str, list[str]]:
    result = collection.get(include=["metadatas"])
    unique: dict[str, set[str]] = {
        "equipment_tags": set(),
        "entity_types": set(),
        "failure_modes": set(),
        "safety_hazards": set(),
    }

    for metadata in result.get("metadatas", []) or []:
        if not metadata:
            continue
        for key in unique:
            value = metadata.get(key, "")
            if isinstance(value, str) and value.strip():
                unique[key].update(part for part in value.split("|") if part.strip())

    return {key: sorted(values) for key, values in unique.items()}

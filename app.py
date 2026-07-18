from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from ingest import (
    get_collection,
    get_openai_client,
    get_unique_entities,
    ingest_sources,
)
from rag_engine import ask_question

load_dotenv()

st.set_page_config(
    page_title="Industrial Knowledge Intelligence Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.stApp {
    background: #0b1220;
    color: #e5eefc;
}
header, footer, #MainMenu {
    visibility: hidden;
}
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }
}
.hero-card, .panel-card {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(10, 16, 28, 0.95));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
}
.source-pill {
    display: inline-block;
    padding: 0.35rem 0.6rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
    border-radius: 999px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(14, 165, 233, 0.25);
    color: #8fd3ff;
    font-size: 0.82rem;
}
.small-muted {
    color: #94a3b8;
    font-size: 0.9rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_api_client() -> OpenAI | None:
    try:
        return get_openai_client()
    except Exception:
        return None


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_tag", "")
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_answer", "")
    st.session_state.setdefault("folder_path", "")
    st.session_state.setdefault("uploaded_files", [])


@st.cache_resource
def cached_collection():
    return get_collection()


def transcribe_audio(audio_file, client: OpenAI) -> str:
    if audio_file is None:
        return ""

    file_name = getattr(audio_file, "name", "voice_input.wav")
    file_bytes = audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file.read()
    buffer = io.BytesIO(file_bytes)
    buffer.name = file_name

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
    )
    return (response.text or "").strip()


def render_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("View Source Documents", expanded=False):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**{index}. {source.get('document_name', 'Unknown')}**")
            tags = source.get("equipment_tags", "")
            if tags:
                st.markdown(f"<span class='source-pill'>{tags}</span>", unsafe_allow_html=True)
            st.caption(f"Distance: {source.get('distance', 'n/a')}")
            st.write(source.get("chunk_text", ""))
            st.divider()


def sidebar_discovered_equipment(collection) -> None:
    st.sidebar.subheader("Discovered Equipment")
    entities = get_unique_entities(collection)
    tags = entities.get("equipment_tags", [])

    if not tags:
        st.sidebar.caption("Upload documents to discover equipment tags.")
    else:
        for tag in tags[:30]:
            if st.sidebar.button(tag, use_container_width=True):
                st.session_state.selected_tag = tag

    st.sidebar.subheader("Extracted Entities")
    for label, values in entities.items():
        if values:
            st.sidebar.markdown(f"**{label.replace('_', ' ').title()}**")
            st.sidebar.write(", ".join(values[:20]))


def process_documents(collection) -> None:
    uploaded_files = st.session_state.get("uploaded_files", [])
    folder_path = st.session_state.get("folder_path", "").strip()
    client = get_api_client()

    sources = []
    if folder_path:
        folder = Path(folder_path)
        if not folder.exists():
            st.sidebar.error(f"Folder not found: {folder}")
            return
        sources.extend([path for path in folder.iterdir() if path.is_file()])

    sources.extend(uploaded_files or [])

    if not sources:
        st.sidebar.warning("Add a folder path or upload files first.")
        return

    with st.spinner("Processing documents and building the industrial knowledge base..."):
        try:
            result = ingest_sources(sources, client=client)
        except Exception as exc:
            st.sidebar.error(f"Document processing failed: {exc}")
            return

    st.sidebar.success(
        f"Processed {len(result['processed_documents'])} document(s) and added {result['chunks_added']} chunk(s)."
    )
    for warning in result["warnings"]:
        st.sidebar.warning(warning)


def ask_and_store(question: str, collection) -> None:
    client = get_api_client()
    if client is None:
        st.error("OPENAI_API_KEY is required for question answering.")
        return

    with st.spinner("Searching industrial knowledge base..."):
        try:
            result = ask_question(question, collection, client=client)
        except Exception as exc:
            st.error(f"Unable to answer the question: {exc}")
            return

    answer = result["answer"]
    st.session_state.last_answer = answer
    st.session_state.last_sources = result["source_documents"]
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    initialize_state()
    collection = cached_collection()

    st.title("Industrial Knowledge Intelligence Platform")
    st.markdown(
        "<div class='hero-card'>"
        "<div class='small-muted'>AI brain for plant documents, logs, and P&amp;IDs</div>"
        "<p>Ask a field question and get cited answers from the current document set.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Ingestion")
        st.session_state["folder_path"] = st.text_input(
            "Folder path",
            value=st.session_state.get("folder_path", ""),
            placeholder="C:\\path\\to\\industrial_docs",
        )
        st.session_state["uploaded_files"] = st.file_uploader(
            "Upload PDF, CSV, Excel, or Image files",
            type=["pdf", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if st.button("Process Documents", use_container_width=True):
            process_documents(collection)
        st.divider()
        sidebar_discovered_equipment(collection)

    if st.session_state.selected_tag:
        st.info(f"Selected equipment tag: {st.session_state.selected_tag}")

    for message in st.session_state.messages:
        render_message(message)

    user_input = st.chat_input("Ask about equipment, failures, safety, or maintenance...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_message(st.session_state.messages[-1])
        ask_and_store(user_input, collection)
        render_message(st.session_state.messages[-1])

    client = get_api_client()
    st.markdown("### Voice input")
    audio_input = getattr(st, "audio_input", None)
    if callable(audio_input) and client is not None:
        audio_data = audio_input("Record a question")
        if audio_data is not None and st.button("Transcribe voice input", use_container_width=True):
            try:
                text = transcribe_audio(audio_data, client)
                if text:
                    st.session_state.messages.append({"role": "user", "content": text})
                    ask_and_store(text, collection)
                    st.rerun()
                else:
                    st.warning("No speech detected in the audio clip.")
            except Exception as exc:
                st.error(f"Voice transcription failed: {exc}")
    else:
        st.caption("Voice input is available when Streamlit exposes st.audio_input and OPENAI_API_KEY is set.")

    if st.session_state.last_answer:
        st.markdown("### Latest Answer")
        st.markdown(st.session_state.last_answer)
        render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()

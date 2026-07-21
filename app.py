from __future__ import annotations

import io
import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timezone
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

ACCESS_DB_PATH = Path("data") / "employee_access.db"

st.set_page_config(
    page_title="Industrial Knowledge Integration Platform",
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
    st.session_state.setdefault("current_employee_id", "")
    st.session_state.setdefault("current_employee_name", "")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _password_hash(password: str) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000).hex()
    return f"{salt}${digest}"


def _password_matches(password: str, stored_password_hash: str) -> bool:
    try:
        salt, expected_digest = stored_password_hash.split("$", maxsplit=1)
    except ValueError:
        return False
    computed_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        100_000,
    ).hex()
    return hmac.compare_digest(computed_digest, expected_digest)


def _get_access_db_connection() -> sqlite3.Connection:
    ACCESS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ACCESS_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_access_db() -> None:
    with _get_access_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                employee_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS login_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                login_at TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_upload_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                chunks_added INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL
            )
            """
        )


def create_employee(employee_id: str, employee_name: str, password: str) -> tuple[bool, str]:
    normalized_id = employee_id.strip()
    normalized_name = employee_name.strip()
    if not normalized_id or not normalized_name or not password:
        return False, "Employee ID, name, and password are required."

    try:
        with _get_access_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO employees (employee_id, employee_name, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_id, normalized_name, _password_hash(password), _utc_now_iso()),
            )
        return True, "Employee account created."
    except sqlite3.IntegrityError:
        return False, f"Employee ID '{normalized_id}' already exists."


def authenticate_employee(employee_id: str, password: str) -> tuple[bool, str]:
    normalized_id = employee_id.strip()
    if not normalized_id or not password:
        return False, ""

    with _get_access_db_connection() as connection:
        row = connection.execute(
            "SELECT employee_name, password_hash FROM employees WHERE employee_id = ?",
            (normalized_id,),
        ).fetchone()

    if row is None:
        return False, ""
    if not _password_matches(password, str(row["password_hash"])):
        return False, ""
    return True, str(row["employee_name"])


def record_login(employee_id: str, status: str) -> None:
    with _get_access_db_connection() as connection:
        connection.execute(
            "INSERT INTO login_records (employee_id, login_at, status) VALUES (?, ?, ?)",
            (employee_id.strip(), _utc_now_iso(), status),
        )


def record_document_uploads(employee_id: str, processed_entries: list[dict[str, str | int]]) -> None:
    if not processed_entries:
        return

    payload = [
        (
            employee_id.strip(),
            str(entry.get("document_name", "")),
            str(entry.get("source_type", "")),
            int(entry.get("chunks_added", 0)),
            _utc_now_iso(),
        )
        for entry in processed_entries
    ]
    with _get_access_db_connection() as connection:
        connection.executemany(
            """
            INSERT INTO document_upload_records (
                employee_id,
                document_name,
                source_type,
                chunks_added,
                uploaded_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )


def get_recent_login_records(employee_id: str, limit: int = 10) -> list[sqlite3.Row]:
    with _get_access_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT employee_id, login_at, status
            FROM login_records
            WHERE employee_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (employee_id.strip(), limit),
        ).fetchall()
    return list(rows)


def get_recent_upload_records(employee_id: str, limit: int = 10) -> list[sqlite3.Row]:
    with _get_access_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT document_name, source_type, chunks_added, uploaded_at
            FROM document_upload_records
            WHERE employee_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (employee_id.strip(), limit),
        ).fetchall()
    return list(rows)


def render_auth_gate() -> bool:
    current_employee_id = st.session_state.get("current_employee_id", "")
    if current_employee_id:
        return True

    st.title("Employee Access")
    st.caption("Login is required to use document ingestion and question answering.")

    login_tab, register_tab = st.tabs(["Login", "Create Employee"])

    with login_tab:
        with st.form("employee-login-form"):
            employee_id = st.text_input("Employee ID", key="login_employee_id").strip()
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            is_valid, employee_name = authenticate_employee(employee_id, password)
            if is_valid:
                st.session_state["current_employee_id"] = employee_id
                st.session_state["current_employee_name"] = employee_name
                record_login(employee_id, "success")
                st.success(f"Welcome, {employee_name}.")
                st.rerun()
            else:
                record_login(employee_id, "failed")
                st.error("Invalid employee credentials.")

    with register_tab:
        with st.form("employee-create-form"):
            new_employee_id = st.text_input("New Employee ID", key="new_employee_id").strip()
            employee_name = st.text_input("Employee Name", key="new_employee_name").strip()
            new_password = st.text_input("Password", type="password", key="new_employee_password")
            created = st.form_submit_button("Create Employee", use_container_width=True)

        if created:
            is_created, message = create_employee(new_employee_id, employee_name, new_password)
            if is_created:
                st.success(message)
            else:
                st.error(message)

    return False


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
    current_employee_id = st.session_state.get("current_employee_id", "").strip()
    if not current_employee_id:
        st.sidebar.error("Login is required before processing documents.")
        return

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
            result = ingest_sources(sources, client=client, uploaded_by=current_employee_id)
        except Exception as exc:
            st.sidebar.error(f"Document processing failed: {exc}")
            return

    record_document_uploads(current_employee_id, result.get("processed_entries", []))
    st.sidebar.success(
        f"Processed {len(result['processed_documents'])} document(s) and added {result['chunks_added']} chunk(s)."
    )
    for warning in result["warnings"]:
        st.sidebar.warning(warning)


def ask_and_store(question: str, collection) -> None:
    client = get_api_client()
    if client is None:
        st.error("Inference Service API Key is required for question answering.")
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
    initialize_access_db()
    initialize_state()
    if not render_auth_gate():
        return
    collection = cached_collection()
    employee_id = st.session_state.get("current_employee_id", "")
    employee_name = st.session_state.get("current_employee_name", "")

    st.title("Industrial Knowledge Integration Platform")
    st.caption(f"Logged in as {employee_name} ({employee_id})")
    st.markdown(
        "<div class='hero-card'>"
        "<div class='small-muted'>Integrated search & compliance verification engine for plant documents, logs, and P&amp;IDs</div>"
        "<p>Ask a field question and get cited answers from the current document set.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        if st.button("Logout", use_container_width=True):
            st.session_state["current_employee_id"] = ""
            st.session_state["current_employee_name"] = ""
            st.rerun()

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

        with st.expander("Recent Login Records", expanded=False):
            for row in get_recent_login_records(employee_id, limit=8):
                st.caption(f"{row['login_at']} | {row['status']}")
        with st.expander("Recent Upload Records", expanded=False):
            for row in get_recent_upload_records(employee_id, limit=8):
                st.caption(
                    f"{row['uploaded_at']} | {row['document_name']} ({row['source_type']}) | chunks: {row['chunks_added']}"
                )
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
        st.caption("Voice input is available when Streamlit exposes st.audio_input and API Key is set.")

    if st.session_state.last_answer:
        st.markdown("### Latest Answer")
        st.markdown(st.session_state.last_answer)
        render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()

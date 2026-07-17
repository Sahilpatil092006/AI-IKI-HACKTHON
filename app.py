import os
import re
import json
import glob
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import chromadb
from chromadb.utils import embedding_functions
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DOCS_FOLDER = "sample_documents"
CHROMA_DIR = ".chroma_db"
COLLECTION_NAME = "industrial_docs"
RULES_FILE = "compliance_rules.json"

# Groq free-tier API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # fast + free tier friendly

st.set_page_config(page_title="Industrial Knowledge Copilot", layout="wide")

# ---------------------------------------------------------------------------
# CUSTOM CSS & THEME INJECTION (Factory Control Room Aesthetic)
# ---------------------------------------------------------------------------
CSS = """
<style>
/* Import High-Tech Fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Page Styling */
.stApp {
    background-color: #0E1117;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

/* Custom Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1.5px;
    color: #00E5FF;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
}

/* Hide Default Streamlit Elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Command Panel / Left Sidebar Card */
.control-panel {
    background: linear-gradient(135deg, #161B22 0%, #0F1319 100%);
    border: 1px solid #30363D;
    border-left: 4px solid #00E5FF;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 229, 255, 0.05);
    margin-bottom: 20px;
}

.control-panel h4 {
    color: #8B949E;
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0;
    margin-bottom: 15px;
}

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 10px;
}

.status-online {
    background-color: #39FF14;
    box-shadow: 0 0 10px #39FF14;
}

.status-offline {
    background-color: #FF5252;
    box-shadow: 0 0 10px #FF5252;
}

/* Navigation tabs container and active styling */
div[data-testid="stHorizontalBlock"] {
    background-color: #0E1117;
}

div[data-testid="stButton"] button {
    background-color: #161B22 !important;
    color: #E2E8F0 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    padding: 10px 16px !important;
    font-weight: 500 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stButton"] button:hover {
    border-color: #00E5FF !important;
    color: #00E5FF !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.25) !important;
    transform: translateY(-1px);
}

.active-tab div[data-testid="stButton"] button {
    border-color: #00E5FF !important;
    color: #00E5FF !important;
    background-color: rgba(0, 229, 255, 0.08) !important;
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.2) !important;
    border-bottom: 3px solid #00E5FF !important;
}

/* Terminal Answer style */
.terminal-response {
    background-color: #161B22;
    border-left: 4px solid #00E5FF;
    border-radius: 6px;
    padding: 20px;
    font-family: 'Inter', sans-serif;
    box-shadow: inset 0 0 15px rgba(0, 229, 255, 0.05), 0 8px 30px rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
    line-height: 1.6;
    border: 1px solid #30363D;
    border-left: 4px solid #00E5FF;
}

/* Evidence Cards */
.evidence-card {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
}

.evidence-card:hover {
    border-color: #00E5FF;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.15);
}

/* Graph custom wrapper styling */
.graph-container {
    border: 1px solid #00E5FF;
    border-radius: 8px;
    background-color: #111111;
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.15);
    padding: 5px;
    margin-bottom: 20px;
}

/* Compliance audit styling */
.compliance-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 14px;
    background-color: #161B22;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.compliance-table th {
    background-color: #0E1117;
    color: #00E5FF;
    font-family: 'Orbitron', sans-serif;
    text-align: left;
    padding: 12px 15px;
    border-bottom: 2px solid #30363D;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
}

.compliance-table td {
    padding: 12px 15px;
    border-bottom: 1px solid #30363D;
    color: #E2E8F0;
}

.compliance-table tr.row-evidence {
    background-color: rgba(57, 255, 20, 0.03);
}

.compliance-table tr.row-evidence:hover {
    background-color: rgba(57, 255, 20, 0.08);
}

.compliance-table tr.row-gap {
    background-color: rgba(255, 82, 82, 0.03);
}

.compliance-table tr.row-gap:hover {
    background-color: rgba(255, 82, 82, 0.08);
}

.badge-evidence {
    background-color: rgba(57, 255, 20, 0.1);
    color: #39FF14;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: bold;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    border: 1px solid rgba(57, 255, 20, 0.3);
    text-transform: uppercase;
}

.badge-gap {
    background-color: rgba(255, 82, 82, 0.1);
    color: #FF5252;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: bold;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    border: 1px solid rgba(255, 82, 82, 0.3);
    text-transform: uppercase;
}

/* Custom styled detail tags */
details summary {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
    outline: none;
}

/* Fixed Bottom query dock styling */
div[data-testid="stForm"] {
    position: fixed;
    bottom: 20px;
    left: 27%;
    width: 70%;
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-top: 2px solid #00E5FF !important;
    border-radius: 8px !important;
    padding: 15px !important;
    z-index: 999;
    box-shadow: 0 -10px 25px rgba(0, 0, 0, 0.6) !important;
}

div[data-testid="stForm"] input {
    background-color: #0E1117 !important;
    color: #FFFFFF !important;
    border: 1px solid #30363D !important;
}

div[data-testid="stForm"] button {
    background-color: #00E5FF !important;
    color: #0E1117 !important;
    font-weight: 700 !important;
    border: none !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
}

div[data-testid="stForm"] button:hover {
    background-color: #00B2CC !important;
    color: #0E1117 !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.5) !important;
}

/* Spacing at bottom of screen */
.bottom-spacer {
    height: 120px;
}
</style>
"""

# ---------------------------------------------------------------------------
# ENTITY EXTRACTION (rule-based, free, no external calls)
# ---------------------------------------------------------------------------
EQUIPMENT_PATTERN = re.compile(r"\b([A-Z]{1,4}-\d{2,4})\b")
REGULATION_PATTERN = re.compile(
    r"\b(OISD-STD-\d+|OISD-\d+|PESO|Factory Act\s?\d{0,4}|SOP-[A-Z]+-\d+)\b"
)
DATE_PATTERN = re.compile(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b")
PERSON_PATTERN = re.compile(
    r"(?:By|Inspector|Supervisor|Technician)\s*[:\-]?\s*([A-Z]\.\s?[A-Za-z]+)"
)


def extract_entities(text):
    equipment = sorted(set(EQUIPMENT_PATTERN.findall(text)))
    regulations = sorted(set(REGULATION_PATTERN.findall(text)))
    dates = sorted(set(DATE_PATTERN.findall(text)))
    people = sorted(set(PERSON_PATTERN.findall(text)))
    return {
        "equipment": equipment,
        "regulations": regulations,
        "dates": dates,
        "people": people,
    }


def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------------------------------
def read_pdf(path):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        return f"[Could not read PDF: {e}]"


@st.cache_resource(show_spinner="Loading & indexing documents (first run downloads a small free embedding model, ~90MB)...")
def load_and_index_documents():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # free, runs locally, no API key
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh collection each app start
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    doc_entities = {}
    all_ids, all_docs, all_meta = [], [], []

    files = sorted(glob.glob(os.path.join(DOCS_FOLDER, "*.txt"))) + sorted(
        glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    )

    for path in files:
        fname = os.path.basename(path)
        if path.endswith(".pdf"):
            text = read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        entities = extract_entities(text)
        doc_entities[fname] = entities

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{fname}::chunk{i}")
            all_docs.append(chunk)
            all_meta.append({"source": fname})

    if all_docs:
        collection.add(ids=all_ids, documents=all_docs, metadatas=all_meta)

    return collection, doc_entities, files


# ---------------------------------------------------------------------------
# RAG QUERY (Groq free LLM)
# ---------------------------------------------------------------------------
def ask_llm_with_context(question, context_chunks):
    context_text = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks]
    )
    prompt = f"""You are an Industrial Knowledge Copilot for a plant operations team.
Answer the question ONLY using the context below. If the answer is not in the
context, say you don't have enough information in the indexed documents.
Always mention which source document(s) you used at the end of your answer.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

    if not GROQ_API_KEY:
        return (
            "⚠️ No GROQ_API_KEY set, so I can't call the free LLM. "
            "Here is the most relevant raw context instead:\n\n" + context_text[:1200]
        )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM call failed ({e}). Raw context:\n\n{context_text[:1200]}"


# ---------------------------------------------------------------------------
# KNOWLEDGE GRAPH
# ---------------------------------------------------------------------------
def build_knowledge_graph(doc_entities):
    G = nx.Graph()
    for doc, ents in doc_entities.items():
        G.add_node(doc, type="document", color="#4C7EF3", size=25)
        for eq in ents["equipment"]:
            G.add_node(eq, type="equipment", color="#F39C4C", size=18)
            G.add_edge(doc, eq)
        for reg in ents["regulations"]:
            G.add_node(reg, type="regulation", color="#4CF39C", size=18)
            G.add_edge(doc, reg)
        for person in ents["people"]:
            G.add_node(person, type="person", color="#C44CF3", size=14)
            G.add_edge(doc, person)
    return G


def render_graph_html(G):
    net = Network(height="550px", width="100%", bgcolor="#111111", font_color="white")
    for node, attrs in G.nodes(data=True):
        net.add_node(node, label=node, color=attrs.get("color", "#888"), size=attrs.get("size", 15))
    for u, v in G.edges():
        net.add_edge(u, v)
    net.repulsion(node_distance=180, spring_length=180)
    path = "knowledge_graph.html"
    net.write_html(path)
    return path


# ---------------------------------------------------------------------------
# COMPLIANCE CHECKER
# ---------------------------------------------------------------------------
def run_compliance_check(all_text_by_doc):
    with open(RULES_FILE, "r") as f:
        rules = json.load(f)["rules"]

    combined_text = " ".join(all_text_by_doc.values()).lower()
    results = []
    for rule in rules:
        found = any(kw.lower() in combined_text for kw in rule["keywords"])
        results.append({
            "Rule": rule["rule_id"],
            "Regulation": rule["regulation"],
            "Requirement": rule["requirement"],
            "Status": "✅ Evidence Found" if found else "⚠️ Gap - No Evidence Found",
        })
    return pd.DataFrame(results)


# Helper function to generate high-tech compliance table
def render_compliance_table(df):
    html = '<table class="compliance-table">'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    
    for _, row in df.iterrows():
        status = row['Status']
        row_class = "row-gap" if "Gap" in status else "row-evidence"
        html += f'<tr class="{row_class}">'
        for col in df.columns:
            val = row[col]
            if col == "Status":
                if "Gap" in status:
                    html += f'<td><span class="badge-gap">{val}</span></td>'
                else:
                    html += f'<td><span class="badge-evidence">{val}</span></td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


# Helper function to render colored entity pills
def make_badges(items, bg_color):
    if not items:
        return '<span style="color: #8B949E; font-style: italic;">None detected</span>'
    badges = []
    for item in items:
        badges.append(
            f'<span style="background-color: {bg_color}; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-size: 11px; margin-right: 6px; display: inline-block; font-weight: bold; font-family: monospace; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{item}</span>'
        )
    return " ".join(badges)


# ---------------------------------------------------------------------------
# STREAMLIT UI LAYOUT & FLOW
# ---------------------------------------------------------------------------
# Inject CSS top-level
st.markdown(CSS, unsafe_allow_html=True)

# Load context data
collection, doc_entities, files = load_and_index_documents()

raw_text_by_doc = {}
for path in files:
    fname = os.path.basename(path)
    if path.endswith(".pdf"):
        raw_text_by_doc[fname] = read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw_text_by_doc[fname] = f.read()

# Setup Session State for navigation & answers
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Copilot"
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""
if "current_context" not in st.session_state:
    st.session_state.current_context = []

# Core Dashboard Columns
col_sidebar, col_main = st.columns([1, 3])

# LEFT COLUMN: Persistent Command Panel
with col_sidebar:
    groq_status_class = "status-online" if GROQ_API_KEY else "status-offline"
    groq_status_text = "ONLINE" if GROQ_API_KEY else "OFFLINE (No Key)"
    
    st.markdown(f"""
    <div class="control-panel">
        <h4>⚡ SYSTEM TELEMETRY</h4>
        <div style="margin-top: 15px;">
            <p style="margin: 8px 0; font-size: 13px;"><span class="status-indicator status-online"></span>CORE ENGINE: ACTIVE</p>
            <p style="margin: 8px 0; font-size: 13px;"><span class="status-indicator status-online"></span>VECTOR DB: CONNECTED</p>
            <p style="margin: 8px 0; font-size: 13px;"><span class="status-indicator status-online"></span>EMBEDDINGS: LOCAL</p>
        </div>
        <hr style="border-color: #30363D; margin: 15px 0;">
        <h4>🌐 API GATEWAY</h4>
        <div style="margin-top: 10px;">
            <p style="margin: 8px 0; font-size: 13px;"><span class="status-indicator {groq_status_class}"></span>GROQ LLM: {groq_status_text}</p>
            <p style="font-size: 11px; color: #8B949E; margin: 4px 0 0 20px;">Model: {GROQ_MODEL}</p>
        </div>
        <hr style="border-color: #30363D; margin: 15px 0;">
        <h4>📊 STORAGE TELEMETRY</h4>
        <div style="margin-top: 10px;">
            <p style="margin: 0; font-size: 28px; font-weight: bold; color: #00E5FF; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);">{len(files)}</p>
            <p style="font-size: 11px; color: #8B949E; margin-top: 4px;">Ingested Documents</p>
        </div>
        <hr style="border-color: #30363D; margin: 15px 0;">
        <h4>⚙️ DOCK CONTROLS</h4>
        <div style="font-size: 11px; color: #8B949E; line-height: 1.4;">
            <p style="margin: 5px 0;">Drop text or PDF files into <code>sample_documents/</code> and restart app to ingest.</p>
            <p style="margin: 10px 0 0 0;"><a href="https://console.groq.com/keys" target="_blank" style="color: #00E5FF; text-decoration: none;">Get Free Groq Key ↗</a></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# RIGHT COLUMN: Main Control Room Dashboard & Tabs
with col_main:
    st.markdown('<h1 style="margin-top: 0; color: #00E5FF;">🏭 FACTORY INTEL COPILOT</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8B949E; margin-bottom: 20px;">Asset & Operations Brain — RAG Copilot + Knowledge Graph + Compliance Gap Checker</p>', unsafe_allow_html=True)

    # High-Tech Segmented Custom Switcher Navigation
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    with t_col1:
        is_active = (st.session_state.active_tab == "Copilot")
        if is_active:
            st.markdown('<div class="active-tab">', unsafe_allow_html=True)
        if st.button("💬 Copilot Panel", use_container_width=True, key="btn_copilot"):
            st.session_state.active_tab = "Copilot"
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    with t_col2:
        is_active = (st.session_state.active_tab == "Knowledge Graph")
        if is_active:
            st.markdown('<div class="active-tab">', unsafe_allow_html=True)
        if st.button("🕸️ Graph Explorer", use_container_width=True, key="btn_kg"):
            st.session_state.active_tab = "Knowledge Graph"
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    with t_col3:
        is_active = (st.session_state.active_tab == "Compliance")
        if is_active:
            st.markdown('<div class="active-tab">', unsafe_allow_html=True)
        if st.button("✅ Compliance Audits", use_container_width=True, key="btn_compliance"):
            st.session_state.active_tab = "Compliance"
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    with t_col4:
        is_active = (st.session_state.active_tab == "Documents")
        if is_active:
            st.markdown('<div class="active-tab">', unsafe_allow_html=True)
        if st.button("📄 Document Index", use_container_width=True, key="btn_docs"):
            st.session_state.active_tab = "Documents"
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #30363D; margin-top: 10px; margin-bottom: 25px;">', unsafe_allow_html=True)

    # --- TAB 1: COPILOT PANEL ---
    if st.session_state.active_tab == "Copilot":
        st.subheader("Query Asset Intelligence")
        st.markdown("<p style='color: #8B949E;'>Search operational guidelines, schedules, equipment logs, and compliance records via local semantic vector matching and LLM generation.</p>", unsafe_allow_html=True)
        
        # Display response if available in state
        if st.session_state.current_answer:
            st.markdown("### Copilot Analysis")
            st.markdown(f"""
            <div class="terminal-response">
                <div style="color: #00E5FF; font-weight: bold; font-family: 'Orbitron', sans-serif; margin-bottom: 10px; font-size: 14px;">
                    🤖 COPILOT COGNITIVE ENGINE - CORE SYSTEM RESPONSE
                </div>
                <div style="font-size: 15px; color: #E2E8F0; font-family: 'Inter', sans-serif;">
                    {st.session_state.current_answer}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔍 Retrieved Evidence Cards")
            for c in st.session_state.current_context:
                st.markdown(f"""
                <div class="evidence-card">
                    <details>
                        <summary style="cursor: pointer; font-family: 'Orbitron', sans-serif; color: #FF9100; font-weight: bold; font-size: 13px;">
                            📄 DOCUMENT REFERENCE: {c['source']} <span style="float: right; color: #00E5FF; font-size: 11px;">CLICK TO EXPAND DATA CHUNK</span>
                        </summary>
                        <div style="margin-top: 10px; color: #8B949E; font-size: 13px; font-family: 'Share Tech Mono', monospace; border-top: 1px solid #30363D; padding-top: 10px;">
                            {c['text']}
                        </div>
                    </details>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)

        # Fixed Bottom Query Form Input
        with st.form(key="query_form", clear_on_submit=False):
            q_col, btn_col = st.columns([5, 1])
            with q_col:
                question = st.text_input("Your question", label_visibility="collapsed", placeholder="Ask a control room query (e.g. 'Why did P-101 fail?')")
            with btn_col:
                ask_btn = st.form_submit_button("RUN COGNITIVE QUERY", use_container_width=True)

        if ask_btn and question:
            with st.spinner("Retrieving relevant context and generating answer..."):
                results = collection.query(query_texts=[question], n_results=4)
                context_chunks = [
                    {"text": doc, "source": meta["source"]}
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0])
                ]
                answer = ask_llm_with_context(question, context_chunks)
                
                # Save to state and rerun to display
                st.session_state.current_question = question
                st.session_state.current_answer = answer
                st.session_state.current_context = context_chunks
                st.rerun()

    # --- TAB 2: KNOWLEDGE GRAPH ---
    elif st.session_state.active_tab == "Knowledge Graph":
        st.subheader("Interactive Entity Relationship Model")
        st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Explore semantic links between documentation, machinery tags, international standards, and operation events.</p>", unsafe_allow_html=True)
        
        G = build_knowledge_graph(doc_entities)
        html_path = render_graph_html(G)
        
        # Render Graph in Styled Glowing Panel
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=570, scrolling=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Custom HTML Legend Mapping
        st.markdown("""
        <div class="legend-panel" style="margin-top: 15px; padding: 15px; background-color: #161B22; border: 1px solid #30363D; border-radius: 6px;">
            <h4 style="margin: 0 0 10px 0; color: #00E5FF; font-size: 12px; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;">🏷️ GRAPH NETWORK LEGEND</h4>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center;"><span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: #4C7EF3; margin-right: 8px;"></span><span style="font-size: 13px; color: #E2E8F0;">Documents</span></div>
                <div style="display: flex; align-items: center;"><span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: #F39C4C; margin-right: 8px;"></span><span style="font-size: 13px; color: #E2E8F0;">Equipment Tags (P-101 etc.)</span></div>
                <div style="display: flex; align-items: center;"><span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: #4CF39C; margin-right: 8px;"></span><span style="font-size: 13px; color: #E2E8F0;">Regulations/SOPs</span></div>
                <div style="display: flex; align-items: center;"><span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: #C44CF3; margin-right: 8px;"></span><span style="font-size: 13px; color: #E2E8F0;">Personnel</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- TAB 3: COMPLIANCE GAP CHECK ---
    elif st.session_state.active_tab == "Compliance":
        st.subheader("Compliance Matrix & GAP Telemetry")
        st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Evaluates current ingested corpus text against core requirements rules. Red rows identify gaps requiring immediate audit attention.</p>", unsafe_allow_html=True)
        
        df = run_compliance_check(raw_text_by_doc)
        
        # Inject styled custom compliance grid
        st.markdown(render_compliance_table(df), unsafe_allow_html=True)
        
        gaps = (df["Status"].str.contains("Gap")).sum()
        if gaps:
            st.markdown(f"""
            <div style="background-color: rgba(255, 82, 82, 0.08); border: 1px solid #FF5252; padding: 15px; border-radius: 6px; margin-top: 20px; box-shadow: 0 0 15px rgba(255,82,82,0.15);">
                <span style="color: #FF5252; font-weight: bold; font-family: 'Orbitron', sans-serif;">⚠️ TELEMETRY WARN:</span> 
                <span style="color: #E2E8F0;">{gaps} regulatory compliance gaps detected. Immediate investigation required.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: rgba(57, 255, 20, 0.08); border: 1px solid #39FF14; padding: 15px; border-radius: 6px; margin-top: 20px; box-shadow: 0 0 15px rgba(57,255,20,0.15);">
                <span style="color: #39FF14; font-weight: bold; font-family: 'Orbitron', sans-serif;">✅ AUDIT PASS:</span> 
                <span style="color: #E2E8F0;">All rule requirements satisfied. Direct evidence matches found in corpus files.</span>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 4: INDEXED DOCUMENTS ---
    elif st.session_state.active_tab == "Documents":
        st.subheader("Ingested Operational Records Database")
        st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Browse details of the raw corpus. Extracted tags, equipment codes, and people are mapped out as colored data badges.</p>", unsafe_allow_html=True)
        
        for fname, ents in doc_entities.items():
            with st.expander(f"📄 {fname}"):
                st.markdown(f"**Equipment Tags:** {make_badges(ents['equipment'], '#FF9100')}", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top: 10px;'>**Regulations/SOPs:** {make_badges(ents['regulations'], '#00C853')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top: 10px;'>**Timeline Markers:** {make_badges(ents['dates'], '#2979FF')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top: 10px;'>**Assigned Personnel:** {make_badges(ents['people'], '#AA00FF')}</div>", unsafe_allow_html=True)

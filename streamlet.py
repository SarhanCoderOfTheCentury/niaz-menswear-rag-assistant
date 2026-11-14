"""Frontend. Run: streamlit run streamlet.py"""
from __future__ import annotations

import json
import os

import streamlit as st

from agent import KnowledgeAgent
from config import Settings
from providers import GeminiProvider, ProviderError
from rag import RAG, load_chunks

st.set_page_config(page_title="Niaz | Knowledge Assistant", page_icon="✳", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="st-"],input,textarea,button {font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;}
.block-container {max-width:1120px;padding-top:2.5rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {border-right:1px solid #e3ebe6;}
[data-testid="stSidebar"] .block-container {padding-top:2rem;}
.eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.16em;color:#52776a;margin-bottom:1.2rem;}
.hero {padding:2rem 0 .9rem;}
.hero h1 {font-size:clamp(2.2rem,5vw,3.5rem);line-height:1.12;font-weight:800;margin:0 0 1rem;}
.hero p {color:#61726b;max-width:660px;font-size:1.05rem;line-height:1.65;}
.brand {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.6rem;letter-spacing:-.05em;}
.brand small {font-family:'DM Sans',sans-serif;display:block;font-size:.69rem;color:#6b7b74;letter-spacing:.14em;font-weight:500;margin-top:.4rem;}
.stamp {display:inline-block;padding:.28rem .65rem;border-radius:20px;background:#e6f0eb;color:#286653;font-size:.72rem;font-weight:600;}
.quiet {font-size:.82rem;color:#718078;line-height:1.6;}
[data-testid="stChatMessage"] {background:white;border:1px solid #e4ebe7;border-radius:16px;padding:1.3rem;margin:1rem 0;}
[data-testid="stMetric"] {background:white;border:1px solid #e4ebe7;border-radius:14px;padding:1rem 1.2rem;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-size:1.6rem;}
.stButton>button {border-radius:10px;min-height:2.8rem;}
[data-testid="stExpander"] {background:white;border-radius:12px;}
@media(max-width:640px){.block-container{padding:1.3rem 1rem 2rem;}.hero{padding-top:1rem;}[data-testid="stChatMessage"]{padding:1rem;}}
</style>""", unsafe_allow_html=True)


def configured_key() -> str:
    value = os.getenv("GEMINI_API_KEY", "")
    try:
        value = st.secrets.get("GEMINI_API_KEY", value)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return value


@st.cache_resource(show_spinner=False)
def make_agent(api_key: str, settings: Settings, fingerprint: str):
    # The fingerprint argument invalidates Streamlit's cache when source files change.
    provider = GeminiProvider(api_key, settings)
    return KnowledgeAgent(RAG(settings, provider), provider)


def render_answer(item: dict):
    with st.chat_message("user", avatar=":material/person:"):
        st.write(item["question"])
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if item["status"] == "conflict":
            st.warning("The evidence contains an unresolved conflict. Check the cited rules.")
        st.markdown(item["answer"])
        st.caption(f"{item['scope'].capitalize()} policy scope · {item['mode'].capitalize()} search · {item['seconds']:.1f}s")
        if item["sources"]:
            with st.expander(f"Sources · {len(item['sources'])} supporting excerpts", expanded=True):
                for n, source in enumerate(item["sources"], 1):
                    meta = source["metadata"]
                    version = f" · v{meta['policy_version']} · {meta['status']}" if meta.get("policy_version") else ""
                    st.write(f"[{n}] {source['filename']}{version}")
                    st.caption(source["excerpt"])
        with st.expander(f"Inspect retrieved chunks · {len(item['retrieved'])}"):
            st.caption("These are the actual chunks sent to Gemini. A retrieved chunk is not automatically a cited source.")
            for i, hit in enumerate(item["retrieved"], 1):
                st.markdown(f"{i}. `{hit['metadata']['source']}` · {hit['metadata']['section']}")
                st.caption(f"Status: {hit['metadata']['status']} · Rank score: {hit.get('score', 0):.4f}")
                st.code(hit["text"], language=None, wrap_lines=True)
            if item.get("validation_note"):
                st.caption(item["validation_note"])


settings = Settings.from_env()
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.markdown('<div class="brand">niaz<span style="color:#528773">.</span><small>STAFF KNOWLEDGE DESK</small></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<span class="stamp">Classroom prototype</span>', unsafe_allow_html=True)
    st.write("")
    st.caption("YOUR WORKSPACE")
    if st.button("New conversation", icon=":material/add:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("KNOWLEDGE BASE")
    try:
        chunks = load_chunks(settings)
        files = sorted({c.metadata["source"] for c in chunks})
        st.write(f"{len(files)} documents available")
        with st.expander("View documents"):
            for filename in files:
                st.caption(filename)
        versions = {(c.metadata["status"], c.metadata["policy_version"], c.metadata["effective_date"])
                    for c in chunks if c.metadata.get("doc_type") == "policy"}
        for status, version, effective in sorted(versions, reverse=True):
            st.caption(f"{status.capitalize()} rules: v{version} · {effective}")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    with st.expander("Search settings"):
        mode = st.selectbox("Retrieval", ["hybrid", "semantic"], format_func=str.capitalize)
        k = st.select_slider("Retrieved chunks", options=[1, 3, 5, 8, 10], value=5)
        scope = st.selectbox("Policy scope", ["auto", "current", "archived", "compare"], format_func=str.capitalize)
        st.caption("Auto uses current rules unless your question asks for old rules, a comparison, or an ISO purchase date.")
    key = configured_key()
    if not key:
        key = st.text_input("Gemini API key", type="password", help="Used for this session. Never added to the repository.")
    rebuild = st.button("Rebuild knowledge index", icon=":material/sync:", use_container_width=True, disabled=not key)
    if st.session_state.messages:
        st.download_button("Export conversation", json.dumps(st.session_state.messages, indent=2),
                           "niaz-conversation.json", "application/json", use_container_width=True)
    st.divider()
    st.markdown('<div class="quiet">Answers from your documents.<br>Sources you can inspect.<br><br>Fictional business data for learning.</div>', unsafe_allow_html=True)

top_left, top_right = st.columns([3, 2])
with top_left:
    st.markdown('<div class="eyebrow">NIAZ MEN’S WEAR / KNOWLEDGE ASSISTANT</div>', unsafe_allow_html=True)
with top_right:
    st.caption("LOCAL CHROMA  ·  POWERED BY GEMINI")
st.markdown('''<div class="hero"><h1>A clear answer.<br>A source behind it.</h1>
<p>Your everyday reference for products, store policies, and procedures.
Ask a question and see the evidence behind the answer.</p></div>''', unsafe_allow_html=True)

metric_cols = st.columns(3)
metric_cols[0].metric("Knowledge documents", len(files))
metric_cols[1].metric("Searchable passages", len(chunks))
metric_cols[2].metric("Policy versions", len({c.metadata.get("policy_version") for c in chunks if c.metadata.get("policy_version")}))
st.write("")

from rag import corpus_fingerprint
fingerprint = corpus_fingerprint(chunks, settings)
agent = make_agent(key, settings, fingerprint) if key else None

if rebuild and agent:
    try:
        with st.spinner("Rebuilding the local knowledge index…"):
            agent.rag.ensure_index(force=True)
        st.success("Index rebuilt from the current documents.")
    except (ProviderError, ValueError, OSError):
        st.error("The index could not be rebuilt. Check your Gemini key, quota, and data files, then try again.")

if not key:
    st.info("Add GEMINI_API_KEY to Streamlit secrets or enter it in the sidebar to build the index and start asking questions.")

pending = None
if not st.session_state.messages:
    st.markdown("Try a question")
    examples = [
        ("Products & prices", "What is the price of the Classic Leather Oxford?"),
        ("Bring facts together", "What does a custom-stitched suit cost, and how much advance do I pay?"),
        ("Policy changes", "Compare the old and current return policies."),
        ("Check what is missing", "Do you deliver internationally?"),
    ]
    for col, (label, question) in zip(st.columns(2), examples[:2]):
        if col.button(label, help=question, use_container_width=True):
            pending = question
    for col, (label, question) in zip(st.columns(2), examples[2:]):
        if col.button(label, help=question, use_container_width=True):
            pending = question

for message in st.session_state.messages:
    render_answer(message)

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Ask your knowledge base", placeholder="e.g. Can a customer return a watch after opening the seal?", max_chars=2000)
    submitted = st.form_submit_button("Ask Niaz", type="primary", icon=":material/arrow_upward:")
if submitted:
    if not question.strip():
        st.warning("Type a question about products, policies, or store procedures to get started.")
    else:
        pending = question
if pending:
    if not agent:
        st.info("Add your Gemini API key in the sidebar or Streamlit secrets first.")
    else:
        try:
            with st.spinner("Finding the evidence and preparing your answer…"):
                result = agent.ask(pending, k=k, mode=mode, scope=scope)
            st.session_state.messages.append(result.to_dict())
            st.rerun()
        except (ProviderError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("The knowledge index could not be read. Use Rebuild knowledge index and try again.")

st.caption("Check cited evidence before using an answer. Missing information is reported explicitly.")

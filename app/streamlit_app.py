"""Streamlit chat UI for the RAGBench multi-domain RAG system.

Run:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import DOMAIN_LABELS, get_config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAGBench RAG", page_icon=":material/support_agent:")


@st.cache_resource(show_spinner="Loading models and index...")
def load_pipeline(domain: str) -> RAGPipeline:
    # embedder / reranker / LLM are cached at module level, so per-domain
    # pipelines share the heavy models and differ only in their collection
    return RAGPipeline(get_config(domain))


# ---------- domain switcher (tab-style) ----------
LABEL_TO_DOMAIN = {v: k for k, v in DOMAIN_LABELS.items()}
selected_label = st.segmented_control(
    "Domain",
    list(LABEL_TO_DOMAIN),
    default="Customer Support",
    label_visibility="collapsed",
)
domain = LABEL_TO_DOMAIN[selected_label or "Customer Support"]

pipeline = load_pipeline(domain)

# one chat history per domain
if "histories" not in st.session_state:
    st.session_state.histories = {}
messages = st.session_state.histories.setdefault(domain, [])

# ---------- sidebar ----------
with st.sidebar:
    st.title(":material/support_agent: RAGBench RAG")
    st.caption(
        "Retrieval-Augmented Generation over the five RAGBench industry "
        "domains. Pick a domain above the chat; each domain has its own "
        "document index and conversation."
    )
    n_chunks = pipeline.retriever.store.count()
    st.metric(f"Indexed chunks ({DOMAIN_LABELS[domain]})", f"{n_chunks:,}")
    if n_chunks == 0:
        st.error(
            f"This domain's index is empty. Run "
            f"`python scripts/build_index.py --domain {domain}` first."
        )
    st.metric("LLM provider", pipeline.generator.provider)

    st.divider()
    top_n = st.slider("Context chunks (after rerank)", 1, 10,
                      pipeline.cfg.rerank_top_n)
    rerank = st.toggle("Cross-encoder reranking", value=True)
    if st.button("Clear conversation", width="stretch"):
        st.session_state.histories[domain] = []
        st.rerun()


def render_sources(contexts: list[dict]):
    with st.expander(f"Sources ({len(contexts)})"):
        for i, c in enumerate(contexts, 1):
            st.markdown(
                f"**[{i}] {c['source']}** - retrieval {c['retrieval_score']:.3f}"
                + (f", rerank {c['rerank_score']:.3f}"
                   if c.get("rerank_score") is not None else "")
            )
            st.caption(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))


# ---------- chat history ----------
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("contexts"):
            render_sources(msg["contexts"])

# ---------- suggestion chips (before first message, per domain) ----------
SUGGESTIONS = {
    "customer_support": {
        ":blue[:material/directions_car:] Uconnect": "What is Uconnect and what can I do with it?",
        ":green[:material/tv:] TV Wi-Fi": "How do I connect my TV to a Wi-Fi network?",
        ":orange[:material/highlight:] Fog lights": "How do I turn on the fog lights?",
    },
    "biomedical": {
        ":blue[:material/coronavirus:] COVID-19": "How does COVID-19 spread between people?",
        ":green[:material/medication:] Treatment": "What treatments have been studied for sepsis?",
    },
    "general_knowledge": {
        ":blue[:material/public:] History": "Who was the first person to walk on the moon?",
        ":green[:material/lightbulb:] Science": "Why is the sky blue?",
    },
    "legal": {
        ":blue[:material/gavel:] Termination": "Under what conditions can this agreement be terminated?",
        ":green[:material/handshake:] Licensing": "What license rights are granted under the agreement?",
    },
    "finance": {
        ":blue[:material/trending_up:] Revenue": "How did the company's revenue change year over year?",
        ":green[:material/payments:] Expenses": "What were the main operating expenses?",
    },
}
prompt = None
if not messages:
    domain_suggestions = SUGGESTIONS.get(domain, {})
    if domain_suggestions:
        selected = st.pills("Try asking:", list(domain_suggestions),
                            label_visibility="collapsed",
                            key=f"pills_{domain}")
        if selected:
            prompt = domain_suggestions[selected]

# ---------- input ----------
typed = st.chat_input(f"Ask a {DOMAIN_LABELS[domain]} question...",
                      submit_mode="disable")
prompt = typed or prompt

if prompt:
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = pipeline.answer(prompt, top_n=top_n, rerank=rerank)
        st.write(result["answer"])
        if result["contexts"]:
            render_sources(result["contexts"])
        st.caption(f"{result['query_type']} - {result['latency_s']}s")

    messages.append({
        "role": "assistant",
        "content": result["answer"],
        "contexts": result["contexts"],
    })

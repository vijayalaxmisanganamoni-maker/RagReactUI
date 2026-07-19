"""Central configuration for the RAGBench multi-domain RAG system."""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = PROJECT_ROOT / ".env"):
    """Tiny .env loader (no dependency): KEY=VALUE lines, no quoting rules."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# RAGBench subsets per industry domain (see the RAGBench paper)
DOMAIN_SUBSETS: dict[str, list[str]] = {
    "customer_support": ["delucionqa", "emanual", "techqa"],
    "biomedical": ["covidqa", "pubmedqa"],
    "general_knowledge": ["hotpotqa", "msmarco", "hagrid", "expertqa"],
    "legal": ["cuad"],
    "finance": ["finqa", "tatqa"],
}

DOMAIN_LABELS: dict[str, str] = {
    "customer_support": "Customer Support",
    "biomedical": "Biomedical Research",
    "general_knowledge": "General Knowledge",
    "legal": "Legal",
    "finance": "Finance",
}

# kept for backwards compatibility
CUSTOMER_SUPPORT_SUBSETS = DOMAIN_SUBSETS["customer_support"]


@dataclass
class Config:
    # ---- domain ----
    domain: str = "customer_support"

    # ---- data ----
    hf_dataset: str = "rungalileo/ragbench"
    subsets: list | None = None          # derived from domain unless overridden
    index_split: str = "train"

    # ---- chunking ----
    chunk_size: int = 1000               # characters
    chunk_overlap: int = 150

    # ---- embeddings / vector store ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_dir: str = str(PROJECT_ROOT / "chroma_db")
    collection_name: str | None = None   # derived from domain unless overridden

    # ---- retrieval ----
    retrieve_k: int = 20                 # candidates fetched from the vector store
    rerank_top_n: int = 5                # kept after cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- generation ----
    # provider: "groq" (needs GROQ_API_KEY) or "local" (flan-t5, CPU-friendly)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("RAG_LLM_PROVIDER", "auto"))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    # local_model: str = "google/flan-t5-base"
    max_new_tokens: int = 512
    temperature: float = 0.1

    # ---- evaluation ----
    # LLM judge for TRACe metrics (scripts/run_eval.py --judge); a stronger
    # model than the generator so its judgments are trustworthy
    judge_model: str = field(
        default_factory=lambda: os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile"))

    def __post_init__(self):
        if self.domain not in DOMAIN_SUBSETS:
            raise ValueError(
                f"Unknown domain {self.domain!r}; choose from {list(DOMAIN_SUBSETS)}")
        if self.subsets is None:
            self.subsets = list(DOMAIN_SUBSETS[self.domain])
        if self.collection_name is None:
            self.collection_name = self.domain


def get_config(domain: str = "customer_support") -> Config:
    return Config(domain=domain)

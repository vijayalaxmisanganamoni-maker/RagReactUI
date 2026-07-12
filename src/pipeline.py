"""End-to-end RAG pipeline:
Query Classification -> Retrieval -> Reranking -> Repacking -> Generation.
"""
import re
import time

from .config import Config, get_config
from .generator import Generator
from .retriever import Retriever

# Small-talk / non-informational queries that don't need retrieval
_SMALLTALK = re.compile(
    r"^\s*(hi|hii+|hello|hey|yo|good\s*(morning|afternoon|evening)|thanks?|"
    r"thank\s*you|ok(ay)?|bye|goodbye|how\s+are\s+you\??)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def classify_query(question: str) -> str:
    """Return 'smalltalk' or 'retrieval'."""
    if _SMALLTALK.match(question or ""):
        return "smalltalk"
    return "retrieval"


class RAGPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.retriever = Retriever(self.cfg)
        self.generator = Generator(self.cfg)

    def answer(self, question: str, k: int | None = None,
               top_n: int | None = None, rerank: bool = True) -> dict:
        t0 = time.time()
        category = classify_query(question)

        if category == "smalltalk":
            return {
                "question": question,
                "query_type": "smalltalk",
                "answer": self.generator.chat(question),
                "contexts": [],
                "latency_s": round(time.time() - t0, 2),
            }

        chunks = self.retriever.retrieve(question, k=k, top_n=top_n, rerank=rerank)
        if not chunks:
            return {
                "question": question,
                "query_type": "retrieval",
                "answer": ("I could not find anything relevant in the support "
                           "documents. The index may be empty - run "
                           "scripts/build_index.py first."),
                "contexts": [],
                "latency_s": round(time.time() - t0, 2),
            }

        packed = self.retriever.repack(chunks)
        answer = self.generator.generate(question, packed)
        return {
            "question": question,
            "query_type": "retrieval",
            "answer": answer,
            "contexts": [
                {
                    "text": c["text"],
                    "source": c["metadata"].get("source"),
                    "parent_doc_id": c["metadata"].get("parent_doc_id"),
                    "retrieval_score": round(c.get("score", 0.0), 4),
                    "rerank_score": round(c["rerank_score"], 4) if "rerank_score" in c else None,
                }
                for c in chunks  # report in relevance order (best first)
            ],
            "latency_s": round(time.time() - t0, 2),
        }

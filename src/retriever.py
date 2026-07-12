"""Retrieval + cross-encoder reranking + repacking.

Follows the pipeline recommended in "Searching for Best Practices in
Retrieval-Augmented Generation" (Wang et al., 2024): dense retrieval of a
candidate pool, rerank with a cross-encoder, then "reverse" repacking so the
most relevant chunk sits closest to the question in the prompt.
"""
from sentence_transformers import CrossEncoder

from .config import Config
from .vector_store import VectorStore

_reranker_cache: dict[str, CrossEncoder] = {}


def get_reranker(cfg: Config) -> CrossEncoder:
    if cfg.reranker_model not in _reranker_cache:
        _reranker_cache[cfg.reranker_model] = CrossEncoder(cfg.reranker_model)
    return _reranker_cache[cfg.reranker_model]


class Retriever:
    def __init__(self, cfg: Config, store: VectorStore | None = None):
        self.cfg = cfg
        self.store = store or VectorStore(cfg)
        self.reranker = get_reranker(cfg)

    def retrieve(self, question: str, k: int | None = None,
                 top_n: int | None = None, rerank: bool = True) -> list[dict]:
        """Return the final context chunks for a question, best first."""
        k = k or self.cfg.retrieve_k
        top_n = top_n or self.cfg.rerank_top_n
        candidates = self.store.query(question, k=k)
        if not candidates:
            return []
        if rerank and len(candidates) > 1:
            pairs = [(question, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_n]

    @staticmethod
    def repack(chunks: list[dict]) -> list[dict]:
        """Reverse repacking: least relevant first, most relevant last
        (closest to the question in the final prompt)."""
        return list(reversed(chunks))

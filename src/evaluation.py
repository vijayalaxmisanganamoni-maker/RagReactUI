"""Evaluation metrics for the RAG system, inspired by RAGBench's TRACe framework.

Implemented without an LLM judge so evaluation is free and reproducible:
  - retrieval_hit_rate : did we retrieve any gold document for the question?
  - retrieval_mrr      : reciprocal rank of the first gold document
  - context_relevance  : mean question<->chunk embedding cosine similarity
  - groundedness       : mean over answer sentences of max cosine similarity
                         to a retrieved chunk (proxy for answer faithfulness)
  - answer_similarity  : embedding cosine similarity(answer, reference answer)
  - answer_rouge_l     : ROUGE-L F1 vs the reference answer
"""
import re

import numpy as np

from .config import Config
from .vector_store import get_embedder


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    return dp[len(b)]


def rouge_l_f1(candidate: str, reference: str) -> float:
    c = re.findall(r"\w+", (candidate or "").lower())
    r = re.findall(r"\w+", (reference or "").lower())
    if not c or not r:
        return 0.0
    lcs = _lcs(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


class RAGEvaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = get_embedder(cfg)

    def _cos(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        embs = self.embedder.encode(texts_a + texts_b, normalize_embeddings=True)
        a, b = embs[: len(texts_a)], embs[len(texts_a):]
        return a @ b.T

    def evaluate_example(self, question: str, answer: str, contexts: list[dict],
                         reference_answer: str, gold_doc_ids: list[str]) -> dict:
        m: dict = {}

        # ---- retrieval quality ----
        retrieved_parents = [c.get("parent_doc_id") for c in contexts]
        gold = set(gold_doc_ids or [])
        hit_rank = next(
            (i + 1 for i, pid in enumerate(retrieved_parents) if pid in gold), None
        )
        m["retrieval_hit"] = 1.0 if hit_rank else 0.0
        m["retrieval_rr"] = 1.0 / hit_rank if hit_rank else 0.0

        # ---- context relevance (question vs retrieved chunks) ----
        chunk_texts = [c["text"] for c in contexts if c.get("text")]
        if chunk_texts:
            sims = self._cos([question], chunk_texts)[0]
            m["context_relevance"] = float(np.mean(sims))
        else:
            m["context_relevance"] = 0.0

        # ---- groundedness (answer sentences supported by context) ----
        ans_sents = _sentences(answer)
        if ans_sents and chunk_texts:
            sims = self._cos(ans_sents, chunk_texts)
            m["groundedness"] = float(np.mean(sims.max(axis=1)))
        else:
            m["groundedness"] = 0.0

        # ---- answer quality vs reference ----
        if reference_answer:
            m["answer_similarity"] = float(self._cos([answer], [reference_answer])[0][0])
            m["answer_rouge_l"] = rouge_l_f1(answer, reference_answer)
        return m

    @staticmethod
    def aggregate(per_example: list[dict]) -> dict:
        keys = sorted({k for m in per_example for k in m})
        agg = {}
        for k in keys:
            vals = [m[k] for m in per_example if k in m and m[k] is not None]
            if vals:
                agg[k] = round(float(np.mean(vals)), 4)
        agg["n_examples"] = len(per_example)
        # convention: mean of retrieval_rr over examples = MRR
        if "retrieval_rr" in agg:
            agg["retrieval_mrr"] = agg.pop("retrieval_rr")
        if "retrieval_hit" in agg:
            agg["retrieval_hit_rate"] = agg.pop("retrieval_hit")
        return agg

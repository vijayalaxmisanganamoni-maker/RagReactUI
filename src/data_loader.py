"""Load the Customer Support subsets of RAGBench from Hugging Face.

Each RAGBench example has (at least):
    question   : str
    documents  : list[str]   retrieved passages used as gold context
    response   : str         reference answer
Plus TRACe annotation scores (relevance, utilization, completeness, adherence)
that we use in evaluation when present.
"""
import hashlib
from typing import Iterator

from datasets import load_dataset

from .config import Config


def doc_id(text: str) -> str:
    """Stable id for a document string (used for dedup across examples)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_subset(cfg: Config, subset: str, split: str):
    return load_dataset(cfg.hf_dataset, subset, split=split)


def iter_corpus_documents(cfg: Config, split: str | None = None,
                          max_docs_per_subset: int | None = None) -> Iterator[dict]:
    """Yield unique documents across the configured subsets.

    `split` may be a single split, comma-separated splits, or "all"
    (train+validation+test). Yields dicts:
    {"id": ..., "text": ..., "source": subset_name}
    """
    split = split or cfg.index_split
    splits = (["train", "validation", "test"] if split == "all"
              else [s.strip() for s in split.split(",") if s.strip()])
    seen: set[str] = set()
    for subset in cfg.subsets:
        count = 0
        for sp in splits:
            ds = load_subset(cfg, subset, sp)
            for ex in ds:
                docs = ex.get("documents") or []
                if isinstance(docs, str):
                    docs = [docs]
                for text in docs:
                    text = (text or "").strip()
                    if not text:
                        continue
                    did = doc_id(text)
                    if did in seen:
                        continue
                    seen.add(did)
                    yield {"id": did, "text": text, "source": subset}
                    count += 1
                if max_docs_per_subset is not None and count >= max_docs_per_subset:
                    break
            if max_docs_per_subset is not None and count >= max_docs_per_subset:
                break


def load_eval_examples(cfg: Config, split: str = "test",
                       max_per_subset: int | None = None) -> list[dict]:
    """Load QA examples (question, gold docs, reference answer) for evaluation."""
    examples = []
    for subset in cfg.subsets:
        ds = load_subset(cfg, subset, split)
        n = 0
        for ex in ds:
            q = (ex.get("question") or "").strip()
            resp = (ex.get("response") or "").strip()
            docs = ex.get("documents") or []
            if isinstance(docs, str):
                docs = [docs]
            if not q or not docs:
                continue
            examples.append({
                "subset": subset,
                "question": q,
                "reference_answer": resp,
                "gold_doc_ids": [doc_id((d or "").strip()) for d in docs if (d or "").strip()],
                "annotations": {
                    "relevance_score": ex.get("relevance_score"),
                    "utilization_score": ex.get("utilization_score"),
                    "completeness_score": ex.get("completeness_score"),
                    "adherence_score": ex.get("adherence_score"),
                },
            })
            n += 1
            if max_per_subset is not None and n >= max_per_subset:
                break
    return examples

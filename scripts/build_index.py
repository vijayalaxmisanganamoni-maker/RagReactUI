"""Download the RAGBench Customer Support subsets, chunk the documents,
embed them and store them in the persistent ChromaDB index.

Usage:
    python scripts/build_index.py                       # full corpus
    python scripts/build_index.py --max-docs 300        # quick subset build
    python scripts/build_index.py --subsets delucionqa  # single subset
    python scripts/build_index.py --reset               # rebuild from scratch
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_document, make_splitter
from src.config import get_config
from src.data_loader import iter_corpus_documents
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="customer_support",
                        help="RAGBench domain: customer_support, biomedical, "
                             "general_knowledge, legal, finance")
    parser.add_argument("--subsets", type=str, default=None,
                        help="comma-separated RAGBench subsets (default: all subsets of the domain)")
    parser.add_argument("--split", type=str, default=None,
                        help="split(s) to index: train, test, train,test or all")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="cap unique documents per subset (for quick experiments)")
    parser.add_argument("--reset", action="store_true", help="drop the existing collection first")
    args = parser.parse_args()

    cfg = get_config(args.domain)
    if args.subsets:
        cfg.subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if args.split:
        cfg.index_split = args.split

    store = VectorStore(cfg)
    if args.reset and store.count() > 0:
        print(f"Resetting collection ({store.count()} chunks)...")
        store.reset()

    splitter = make_splitter(cfg)
    t0 = time.time()
    n_docs = n_chunks = 0
    buffer = []

    print(f"Indexing domain={cfg.domain} subsets={cfg.subsets} "
          f"(split={cfg.index_split}, max_docs_per_subset={args.max_docs}, "
          f"collection={cfg.collection_name})")
    for doc in iter_corpus_documents(cfg, max_docs_per_subset=args.max_docs):
        buffer.extend(chunk_document(splitter, doc))
        n_docs += 1
        if len(buffer) >= 512:
            store.add_chunks(buffer)
            n_chunks += len(buffer)
            buffer = []
            print(f"  {n_docs} docs -> {n_chunks} chunks "
                  f"({time.time() - t0:.0f}s elapsed)")
    if buffer:
        store.add_chunks(buffer)
        n_chunks += len(buffer)

    print(f"Done: {n_docs} unique documents, {n_chunks} chunks added, "
          f"collection now has {store.count()} chunks "
          f"({time.time() - t0:.0f}s).")


if __name__ == "__main__":
    main()

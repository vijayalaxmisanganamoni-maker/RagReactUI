"""ChromaDB persistent vector store with sentence-transformers embeddings."""
import chromadb
from sentence_transformers import SentenceTransformer

from .config import Config

_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(cfg: Config) -> SentenceTransformer:
    if cfg.embedding_model not in _embedder_cache:
        _embedder_cache[cfg.embedding_model] = SentenceTransformer(cfg.embedding_model)
    return _embedder_cache[cfg.embedding_model]


class VectorStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(path=cfg.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = get_embedder(cfg)

    def _refresh(self):
        """Re-acquire the collection (a rebuild may have dropped/recreated it)."""
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            self._refresh()
            return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.cfg.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], batch_size: int = 256):
        """Embed and upsert chunk records ({id, text, metadata}).

        Ids are text hashes, so the same boilerplate chunk can appear several
        times in one call - chroma rejects duplicate ids in a single upsert,
        so keep only the first occurrence.
        """
        seen: set[str] = set()
        chunks = [c for c in chunks
                  if not (c["id"] in seen or seen.add(c["id"]))]
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.encode(
                texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
            )
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[c["metadata"] for c in batch],
            )

    def query(self, question: str, k: int | None = None) -> list[dict]:
        """Return top-k chunks: [{id, text, metadata, score}] (score = cosine sim)."""
        k = k or self.cfg.retrieve_k
        if self.count() == 0:  # count() also refreshes a stale collection handle
            return []
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        res = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({
                "id": cid,
                "text": text,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine distance -> similarity
            })
        return hits

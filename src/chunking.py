"""Split documents into overlapping chunks for indexing."""
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config


def make_splitter(cfg: Config) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(splitter: RecursiveCharacterTextSplitter, doc: dict) -> list[dict]:
    """Chunk one corpus document (dict with id/text/source) into chunk records."""
    chunks = []
    for i, piece in enumerate(splitter.split_text(doc["text"])):
        piece = piece.strip()
        if len(piece) < 30:  # drop tiny fragments
            continue
        # id by chunk TEXT hash: identical boilerplate appearing in many docs
        # (e.g. legal disclaimers) collapses to one entry instead of
        # crowding out diverse results at query time
        cid = hashlib.sha1(piece.encode("utf-8")).hexdigest()
        chunks.append({
            "id": cid,
            "text": piece,
            "metadata": {
                "parent_doc_id": doc["id"],
                "source": doc["source"],
                "chunk_index": i,
            },
        })
    return chunks

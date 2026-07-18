"""FastAPI service exposing the Customer Support RAG pipeline.

Run:
    uvicorn app.api:app --reload
Then POST to http://127.0.0.1:8000/ask  {"question": "..."}
Interactive docs at http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import DOMAIN_SUBSETS, get_config
from src.pipeline import RAGPipeline

app = FastAPI(
    title="RAGBench Multi-Domain RAG API",
    description="Retrieval-Augmented Generation over the five RAGBench domains: "
                "customer_support, biomedical, general_knowledge, legal, finance.",
    version="2.0.0",
)

# allow the React dev server (npm run dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipelines: dict[str, RAGPipeline] = {}


def get_pipeline(domain: str = "customer_support") -> RAGPipeline:
    if domain not in _pipelines:
        _pipelines[domain] = RAGPipeline(get_config(domain))
    return _pipelines[domain]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How do I reset my TV to factory settings?"])
    domain: str = Field("customer_support",
                        pattern="^(" + "|".join(DOMAIN_SUBSETS) + ")$")
    top_n: int | None = Field(None, ge=1, le=20, description="context chunks to use")
    rerank: bool = True


class ContextChunk(BaseModel):
    text: str
    source: str | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    question: str
    query_type: str
    answer: str
    contexts: list[ContextChunk]
    latency_s: float


@app.get("/health")
def health():
    domains = {}
    for d in DOMAIN_SUBSETS:
        try:
            domains[d] = get_pipeline(d).retriever.store.count()
        except Exception as e:  # collection missing etc.
            domains[d] = f"unavailable: {e}"
    return {
        "status": "ok",
        "index_chunks_per_domain": domains,
        "llm_provider": get_pipeline().generator.provider,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = get_pipeline(req.domain).answer(
        req.question, top_n=req.top_n, rerank=req.rerank)
    return result


# serve the built React UI (frontend/dist) at / when it exists;
# API routes above keep precedence over the static mount
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")

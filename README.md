# Real World RAG System — Customer Support Domain

Capstone Project 2 (AIML-PGCP): a Retrieval-Augmented Generation pipeline that
answers customer-support questions over the **Customer Support** domain of the
[RAGBench](https://huggingface.co/datasets/rungalileo/ragbench) benchmark:

| RAGBench subset | Corpus |
|---|---|
| `delucionqa` | Jeep vehicle owner-manual QA |
| `emanual` | TV / consumer-electronics user manuals |
| `techqa` | IBM technical-support documents |

## Architecture

The pipeline follows the workflow prescribed in the project brief
(and in *Searching for Best Practices in RAG*, Wang et al. 2024):

```
User query
   |
   v
Query Classification   (small talk vs. retrieval question)
   |
   v
Retrieval              (all-MiniLM-L6-v2 embeddings, ChromaDB, top-20 candidates)
   |
   v
Reranking              (cross-encoder/ms-marco-MiniLM-L-6-v2, keep top-5)
   |
   v
Repacking              (reverse order: most relevant chunk closest to question)
   |
   v
Generation             (Groq Llama-3.3-70B if GROQ_API_KEY set, else local flan-t5-base)
   |
   v
Grounded answer + cited source passages
```

## Project layout

```
src/
  config.py        central configuration
  data_loader.py   RAGBench loading + document dedup
  chunking.py      recursive character chunking (1000 chars, 150 overlap)
  vector_store.py  ChromaDB persistent store + embeddings
  retriever.py     dense retrieval + cross-encoder rerank + repacking
  generator.py     Groq / local flan-t5 answer generation
  pipeline.py      end-to-end RAG pipeline
  evaluation.py    RAGBench-style metrics (no LLM judge needed)
scripts/
  build_index.py   download data, chunk, embed, index
  run_eval.py      evaluate on the test split -> eval_results.json
app/
  api.py           FastAPI service (POST /ask)
  streamlit_app.py Streamlit chat UI
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the index (start small; drop --max-docs for the full corpus)
python scripts/build_index.py --max-docs 300

# 2. (optional, recommended) use Groq for generation
#    set GROQ_API_KEY=<your key>          (Windows)
#    export GROQ_API_KEY=<your key>       (Linux/Mac)

# 3. Chat UI
streamlit run app/streamlit_app.py

# 4. REST API
uvicorn app.api:app --reload
#    -> POST http://127.0.0.1:8000/ask   {"question": "..."}
#    -> docs at http://127.0.0.1:8000/docs

# 5. Evaluation on the RAGBench test split
python scripts/run_eval.py --n 20

# 6. Evaluation with the LLM judge (TRACe metrics, needs GROQ_API_KEY)
python scripts/run_eval.py --n 20 --judge
```

## Evaluation metrics

`scripts/run_eval.py` reports, per question and aggregated, two families of
metrics inspired by RAGBench's TRACe framework.

Judge-free (cheap, reproducible, always computed):

- **retrieval_hit_rate / retrieval_mrr** — is a gold document among the
  retrieved chunks, and how high is it ranked
- **context_relevance** — mean question↔chunk embedding similarity
- **groundedness** — how well each answer sentence is supported by the
  retrieved context (proxy for answer faithfulness / adherence)
- **answer_similarity / answer_rouge_l** — agreement with the RAGBench
  reference answer

LLM judge (`--judge`, uses `llama-3.3-70b-versatile` on Groq — override with
`GROQ_JUDGE_MODEL`): the judge sees the question, the retrieved context and
the answer with sentence keys ('0a', '0b' / 'a', 'b') and returns a
structured judgment (RAGBench's judge prompt), from which the four TRACe
metrics are derived:

- **judge_relevance** — fraction of context sentences relevant to the question
- **judge_utilization** — fraction of context sentences used in the answer
- **judge_completeness** — fraction of relevant sentences that were used
- **judge_adherence** — 1 iff every answer sentence is fully supported

Both families are compared against RAGBench's human/GPT-4 TRACe annotations
via `rmse_*` entries in the summary.

Results are saved to `eval_results_<domain>.json` for the technical report.

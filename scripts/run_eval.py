"""Evaluate the RAG pipeline on RAGBench Customer Support test questions.

Usage:
    python scripts/run_eval.py --n 20                 # 20 questions per subset
    python scripts/run_eval.py --subsets delucionqa --n 50
Results are printed and written to eval_results.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data_loader import load_eval_examples
from src.evaluation import RAGEvaluator
from src.pipeline import RAGPipeline

OUT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="customer_support")
    parser.add_argument("--subsets", type=str, default=None)
    # default to train: with a train-only index, only train questions have
    # their gold documents in the corpus (pass --split test to override)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n", type=int, default=10, help="questions per subset")
    args = parser.parse_args()

    cfg = get_config(args.domain)
    if args.subsets:
        cfg.subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    examples = load_eval_examples(cfg, split=args.split, max_per_subset=args.n)
    print(f"Evaluating {len(examples)} questions from {cfg.subsets} ({args.split} split)")

    pipeline = RAGPipeline(cfg)
    evaluator = RAGEvaluator(cfg)

    per_example, records = [], []
    for i, ex in enumerate(examples, 1):
        result = pipeline.answer(ex["question"])
        metrics = evaluator.evaluate_example(
            question=ex["question"],
            answer=result["answer"],
            contexts=result["contexts"],
            reference_answer=ex["reference_answer"],
            gold_doc_ids=ex["gold_doc_ids"],
        )
        per_example.append(metrics)
        records.append({
            "subset": ex["subset"],
            "question": ex["question"],
            "answer": result["answer"],
            "reference_answer": ex["reference_answer"],
            "metrics": metrics,
            "latency_s": result["latency_s"],
        })
        print(f"[{i}/{len(examples)}] hit={metrics['retrieval_hit']:.0f} "
              f"grounded={metrics['groundedness']:.2f} "
              f"rougeL={metrics.get('answer_rouge_l', 0):.2f} "
              f"({result['latency_s']}s) {ex['question'][:60]}")

    summary = evaluator.aggregate(per_example)
    print("\n===== Aggregate metrics =====")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")

    out_path = OUT_DIR / f"eval_results_{cfg.domain}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "examples": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()

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
from src.evaluation import RAGEvaluator, rmse
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
    parser.add_argument("--judge", action="store_true",
                        help="also score each example with the LLM judge "
                             "(TRACe metrics via Groq, needs GROQ_API_KEY)")
    args = parser.parse_args()

    cfg = get_config(args.domain)
    if args.subsets:
        cfg.subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    examples = load_eval_examples(cfg, split=args.split, max_per_subset=args.n)
    print(f"Evaluating {len(examples)} questions from {cfg.subsets} ({args.split} split)")

    pipeline = RAGPipeline(cfg)
    evaluator = RAGEvaluator(cfg)
    judge = None
    if args.judge:
        from src.judge import LLMJudge
        judge = LLMJudge(cfg)
        print(f"LLM judge enabled ({cfg.judge_model})")

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
        record = {
            "subset": ex["subset"],
            "question": ex["question"],
            "answer": result["answer"],
            "reference_answer": ex["reference_answer"],
            "metrics": metrics,
            "annotations": ex["annotations"],
            "latency_s": result["latency_s"],
        }
        if judge:
            try:
                judged = judge.judge(
                    ex["question"],
                    [c["text"] for c in result["contexts"] if c.get("text")],
                    result["answer"],
                )
                metrics.update(judged["metrics"])
                record["judgment"] = judged["judgment"]
            except Exception as e:
                print(f"  ! judge failed on example {i}: {e}")
        per_example.append(metrics)
        records.append(record)
        line = (f"[{i}/{len(examples)}] hit={metrics['retrieval_hit']:.0f} "
                f"grounded={metrics['groundedness']:.2f} "
                f"rougeL={metrics.get('answer_rouge_l', 0):.2f}")
        if "judge_adherence" in metrics:
            line += f" adherent={metrics['judge_adherence']:.0f}"
        print(f"{line} ({result['latency_s']}s) {ex['question'][:60]}")

    summary = evaluator.aggregate(per_example)

    # RMSE of our predicted scores vs RAGBench TRACe annotations; rmse()
    # drops examples where the annotation is missing (None/NaN)
    annotations = [ex["annotations"] for ex in examples]
    trace_rmse = {
        "rmse_context_relevance": rmse(
            [a["relevance_score"] for a in annotations],
            [m.get("context_relevance") for m in per_example],
        ),
        "rmse_adherence": rmse(
            [a["adherence_score"] for a in annotations],
            [m.get("groundedness") for m in per_example],
        ),
    }
    if judge:
        # judge predictions vs the same annotations (note: annotations score
        # RAGBench's gold documents, the judge scores our retrieved chunks)
        for name, ann_key in [("relevance", "relevance_score"),
                              ("utilization", "utilization_score"),
                              ("completeness", "completeness_score"),
                              ("adherence", "adherence_score")]:
            trace_rmse[f"rmse_judge_{name}"] = rmse(
                [a[ann_key] for a in annotations],
                [m.get(f"judge_{name}") for m in per_example],
            )
    summary.update({k: round(v, 4) for k, v in trace_rmse.items() if v is not None})

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

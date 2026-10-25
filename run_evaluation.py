#!/usr/bin/env python3
"""
Golden-set evaluation harness.

Runs every question in golden_eval_set_2.csv through the REAL pipeline
(generate_answer.answer_question, i.e. the same retrieval + generation code
the website uses) and scores it automatically, category by category.

This has NOT been run against your real corpus/embeddings - that data
only exists on your machine. Running it there is what turns the
"Evaluation Methodology" slide into an actual "Results" slide: this script
writes eval_results_detail.csv and eval_summary.json, and
build_results_chart.py turns those into a chart you can paste straight in.

Usage (from inside the ayurveda-rag folder, with your corpus/index built):
    python run_evaluation.py
    python run_evaluation.py --csv golden_eval_set_2.csv --limit 20   # quick smoke test
    python run_evaluation.py --llm-judge                              # also LLM-grades in-domain answers (slower, costs API calls)

Output:
    eval_results_detail.csv   one row per question: category, verdict, score, latency, answer
    eval_summary.json         category-level pass rates + overall numbers
"""
import argparse
import csv
import json
import re
import time

# ---------------------------------------------------------------------------
# Scoring rules per category. These are intentionally simple and auditable -
# each one checks something the system is actually supposed to do, not a
# vague "did it sound right".
# ---------------------------------------------------------------------------
DECLINE_PHRASES = ["does not contain sufficient evidence", "outside the scope",
                   "does not cover", "cannot answer", "corpus does not contain"]


def is_decline(answer_text):
    low = (answer_text or "").lower()
    return any(p in low for p in DECLINE_PHRASES)


def has_citation(answer_text):
    """Looks for the "(Source: X, page Y)" pattern the system prompt requires."""
    return bool(re.search(r"\(Source:.*?page", answer_text or "", re.IGNORECASE))


def score_row(row, result):
    """Returns (verdict, note) - verdict is one of PASS / FAIL / REVIEW.
    REVIEW means "automated scoring can't judge this one, a human should
    read it" - used for anything that needs actual reasoning/fact-checking,
    which this script deliberately does NOT fake."""
    category = row["category"]
    answer = result.get("answer", "")
    declined = is_decline(answer)
    cited = has_citation(answer)

    if category == "out_of_domain_guardrail":
        # This question should NOT be answerable from the corpus - the
        # correct, verifiable behaviour is declining.
        return ("PASS", "correctly declined") if declined else ("FAIL", "answered when it should have declined")

    if category in ("in_domain", "topic_question"):
        if declined:
            return "FAIL", "declined an in-domain question it should answer"
        if not cited:
            return "FAIL", "answered but with no citation"
        return "REVIEW", "answered with citation - needs a human/LLM check against expected_answer"

    if category == "cornerstone_reasoning":
        # Adversarial/contradiction questions: correctness needs judgement,
        # but citing its reasoning (or correctly declining) is the
        # auto-checkable floor.
        if declined:
            return "REVIEW", "declined - check manually whether that's the right call"
        return "REVIEW", ("cited" if cited else "answered without citation") + " - needs human read for reasoning quality"

    if category in ("multilingual_sanskrit", "multilingual_arabic_unani"):
        if declined:
            return "FAIL", "declined a multilingual in-domain question"
        return "REVIEW", "answered - check the reply is in the right language and cited"

    if category in ("adjacent_domain_tcm", "modern_correlation"):
        return "REVIEW", "needs a human read - judgement call on adjacent-domain framing"

    return "REVIEW", "uncategorised - needs a human read"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="golden_eval_set_2.csv")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N rows (quick smoke test)")
    ap.add_argument("--model", default=None, help="override the model id (default: llm_providers.DEFAULT_MODEL)")
    ap.add_argument("--llm-judge", action="store_true",
                    help="for REVIEW-tagged in-domain answers, ask an LLM to grade similarity "
                         "to expected_answer on a 0-1 scale. Extra API calls - off by default.")
    ap.add_argument("--out-detail", default="eval_results_detail.csv")
    ap.add_argument("--out-summary", default="eval_summary.json")
    args = ap.parse_args()

    from generate_answer import answer_question, MODEL_NAME
    model = args.model or MODEL_NAME

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    judge = None
    if args.llm_judge:
        import llm_providers
        JUDGE_PROMPT = (
            "Score how well ANSWER covers the same facts as EXPECTED on a scale of "
            "0.0 to 1.0 (1.0 = fully consistent, 0.0 = contradicts or misses the point). "
            "Reply with ONLY the number."
        )

        def judge(expected, answer):
            try:
                text = llm_providers.generate(
                    model, JUDGE_PROMPT,
                    f"EXPECTED:\n{expected}\n\nANSWER:\n{answer}", max_tokens=10)
                return float(re.search(r"[01](\.\d+)?", text).group())
            except Exception:
                return None

    detail_rows = []
    print(f"Running {len(rows)} questions through answer_question() with model={model}...")
    for i, row in enumerate(rows, 1):
        t0 = time.time()
        try:
            result = answer_question(row["question"], model=model)
        except Exception as e:
            result = {"answer": f"ERROR: {e}", "sources": [], "retrieval_scores": []}
        latency = time.time() - t0

        verdict, note = score_row(row, result)
        judge_score = ""
        if judge and verdict == "REVIEW" and row["category"] in ("in_domain", "topic_question") and row.get("expected_answer"):
            s = judge(row["expected_answer"], result.get("answer", ""))
            if s is not None:
                judge_score = s
                verdict = "PASS" if s >= 0.5 else "FAIL"
                note = f"LLM-judge similarity={s:.2f}"

        top_score = result.get("retrieval_scores", [None])[0] if result.get("retrieval_scores") else None
        detail_rows.append({
            "id": row.get("id"), "category": row["category"], "question": row["question"],
            "verdict": verdict, "note": note, "judge_score": judge_score,
            "top_retrieval_score": top_score, "declined": is_decline(result.get("answer", "")),
            "cited": has_citation(result.get("answer", "")), "latency_s": round(latency, 2),
            "answer": (result.get("answer") or "")[:500],
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    with open(args.out_detail, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    # ---- summary ----
    by_cat = {}
    for r in detail_rows:
        c = by_cat.setdefault(r["category"], {"PASS": 0, "FAIL": 0, "REVIEW": 0, "total": 0,
                                               "avg_latency_s": 0.0, "cited_rate": 0.0})
        c[r["verdict"]] += 1
        c["total"] += 1
        c["avg_latency_s"] += r["latency_s"]
        c["cited_rate"] += 1 if r["cited"] else 0
    for c in by_cat.values():
        c["avg_latency_s"] = round(c["avg_latency_s"] / c["total"], 2)
        c["cited_rate"] = round(c["cited_rate"] / c["total"], 3)

    summary = {
        "model": model, "total_questions": len(detail_rows),
        "overall": {
            "PASS": sum(r["verdict"] == "PASS" for r in detail_rows),
            "FAIL": sum(r["verdict"] == "FAIL" for r in detail_rows),
            "REVIEW": sum(r["verdict"] == "REVIEW" for r in detail_rows),
        },
        "by_category": by_cat,
        "note": ("REVIEW means automated scoring can't judge correctness - a human (or "
                "--llm-judge) needs to read the answer. Only guardrail/citation/decline "
                "checks are auto-graded PASS/FAIL - that's deliberate, not a bug."),
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {args.out_detail} and {args.out_summary}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()

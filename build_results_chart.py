#!/usr/bin/env python3
"""
Turns eval_summary.json (written by run_evaluation.py) into a chart you can
paste straight into the Results slide. Matches the deck's palette so it
doesn't look bolted on.

Usage:
    python run_evaluation.py              # writes eval_summary.json
    python build_results_chart.py         # writes results_chart.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, GOLD, MUTED, OUTLINE = "#1F4D2B", "#B7791F", "#6B6457", "#C9B48A"

with open("eval_summary.json", encoding="utf-8") as f:
    summary = json.load(f)

cats = summary["by_category"]
order = ["in_domain", "topic_question", "out_of_domain_guardrail", "cornerstone_reasoning",
        "multilingual_sanskrit", "multilingual_arabic_unani", "adjacent_domain_tcm", "modern_correlation"]
labels_map = {
    "in_domain": "In-domain", "topic_question": "Topic questions",
    "out_of_domain_guardrail": "Should decline", "cornerstone_reasoning": "Adversarial reasoning",
    "multilingual_sanskrit": "Sanskrit", "multilingual_arabic_unani": "Arabic/Unani",
    "adjacent_domain_tcm": "Adjacent (TCM)", "modern_correlation": "Modern correlation",
}
present = [c for c in order if c in cats]

fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
fig.patch.set_alpha(0)
ax.set_facecolor("none")

x = range(len(present))
pass_vals = [cats[c]["PASS"] for c in present]
fail_vals = [cats[c]["FAIL"] for c in present]
review_vals = [cats[c]["REVIEW"] for c in present]

ax.bar(x, pass_vals, color=GREEN, label="Pass (auto-graded)")
ax.bar(x, fail_vals, bottom=pass_vals, color="#B3463C", label="Fail (auto-graded)")
bottoms = [p + f for p, f in zip(pass_vals, fail_vals)]
ax.bar(x, review_vals, bottom=bottoms, color=OUTLINE, label="Needs human/LLM review")

ax.set_xticks(list(x))
ax.set_xticklabels([labels_map.get(c, c) for c in present], rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Questions", color=MUTED)
ax.legend(frameon=False, loc="upper right", fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(OUTLINE)
ax.spines["left"].set_color(OUTLINE)
ax.grid(axis="y", color=OUTLINE, alpha=0.3, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("results_chart.png", transparent=True)
print("Wrote results_chart.png")
print(f"Overall: {summary['overall']}")

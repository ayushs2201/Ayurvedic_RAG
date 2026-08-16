# Ayurveda RAG — Test-Driven Retrieval-Augmented Generation for Ayurvedic Knowledge

**Group 22** · RMIT University — Data Science Project (Test-Driven RAG)

> ⚠️ **Disclaimer:** This project is an academic prototype for research and educational purposes only. It does not provide medical advice and is not a substitute for consultation with a qualified healthcare professional.

---

## Overview

Ayurveda is one of the world's oldest documented health systems — a natural, herbal, evidence-informed practice spanning diet, herbs, and lifestyle balance. Despite its global relevance, it remains largely unexplored as a domain for modern NLP and RAG research.

This project builds a **grounded, retrieval-augmented assistant** over a curated Ayurvedic knowledge base, paired with a **test-driven evaluation framework** that measures whether the system's answers are effective, faithful, correctly attributed, and safe — not just whether it produces plausible-sounding text.

## Motivation

- **Underexplored domain:** Few RAG projects touch traditional/complementary medicine — room for original contribution rather than a saturated use case.
- **Real tension for faithfulness testing:** The corpus mixes classical remedies with modern evidence reviews that explicitly hedge ("no clinical evidence yet") — ideal ground truth for testing whether a RAG system preserves uncertainty instead of overclaiming.
- **Global, low-cost relevance:** Practised and studied on every continent; guidance is largely diet-, herb-, and lifestyle-based rather than dependent on costly interventions.
- **Real stakeholders:** health-conscious individuals, Ayurvedic students/practitioners, researchers, and the herbal/wellness industry.

## Dataset

**Source:** [Ayurveda Texts (English)](https://www.kaggle.com/datasets/rcratos/ayurveda-texts-english) — Kaggle, ~634 MB (v1)

| Tier | Contents |
|---|---|
| **Books & treatises** | 20+ full-length texts — classical works (*Charaka Samhita*, *Sushruta Samhita*, *Ashtanga Hridaya*) and modern practitioner guides (*Evidence-Based Ayurveda*, *Everyday Ayurveda*) |
| **Evidence-based articles** | 2,000+ short web-sourced articles — largely institutional/government health guidance (e.g. AIIA) and evidence reviews on specific herbs and compounds |

**Planned extensions** (stretch goals):
- Translated regional-language primary sources (Sanskrit/Hindi)
- Custom web-scraped articles targeting evidence gaps (e.g. WHO traditional-medicine guidance)

## Architecture

```
Data Sources → Preprocessing → Embedding → Vector Search → LLM Generation → Grounded Answer
   (Kaggle +      (clean,      (Sentence-    (semantic       (local LLM,      (cited,
   scraped)      chunk text)   Transformer)   top-k)          via Ollama)      hedged)
                                                                    ↑___________________|
                                                              evaluation feedback loop
```

1. **Data sources** — Kaggle corpus plus optional supplementary sources (see above).
2. **Preprocessing** — clean, de-duplicate, and chunk documents (with overlap) for retrieval.
3. **Embedding** — encode each chunk into a vector using a SentenceTransformer model.
4. **Vector search** — embed the user query with the same model, rank all chunks by cosine similarity, and retrieve the top-k (default k=5).
5. **LLM generation** — a free, locally-hosted LLM (via [Ollama](https://ollama.com/)) generates an answer grounded only in the retrieved chunks.
6. **Grounded answer** — the response is returned with source attribution and is scored against the evaluation framework below; results feed back into retrieval/prompt tuning.

## Evaluation Framework

| Dimension | What we measure |
|---|---|
| **Effectiveness** | % of unanswered / low-confidence questions |
| **Faithfulness** | Whether the answer preserves hedged or uncertain claims rather than overclaiming |
| **Source-attribution correctness** | Whether the cited tier (classical vs. evidence-based) is correct |
| **Fairness / safety** | Whether the system avoids unsafe or misleading medical guidance |

## Tech Stack

- **Embeddings:** SentenceTransformer
- **Vector store:** TBD (e.g. FAISS / Chroma)
- **LLM:** Local, no-cost model via Ollama
- **Language:** Python

## Project Status

🚧 **Proposal stage** — dataset selected, architecture and evaluation framework defined. Link to Trello Board : https://trello.com/invite/b/6a7ab8ac46005f3fdbdf2bd0/ATTIff6d6cbebf520ef6f04e9821bd3756826027CA35/case-studies-wil-group-22

## Roadmap

- [ ] Finalise curated document subset
- [ ] Build baseline RAG pipeline
- [ ] Build test-question set + evaluation harness
- [ ] Tune retrieval & prompting against evaluation results
- [ ] Package demo app and final report

## Team

Group 22 — RMIT University, Master of Data Science (MC267)

## License

Academic project — not licensed for production or clinical use.

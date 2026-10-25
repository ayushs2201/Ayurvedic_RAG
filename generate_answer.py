"""Generation step: hybrid retrieval -> citation-aware prompt -> selected LLM.
Your original logic, now model-agnostic: the model is chosen per request
from llm_providers.MODELS instead of being hard-coded.

Retrieval now goes through query_translation.multi_query_hybrid_search instead
of calling hybrid_search directly - it searches the question as typed AND a
Sanskrit/Ayurvedic-terminology variant (see query_translation.py), so a plain
English query like "cold shoulder" can still find a chunk that only ever says
"Amavata" or "skandha"."""
from query_translation import multi_query_hybrid_search
import llm_providers

MODEL_NAME = llm_providers.DEFAULT_MODEL
MIN_RELEVANCE_SCORE = 0.35   # tune against your golden_eval_set.csv
KEYWORD_WEIGHT = 0.2

SYSTEM_PROMPT = """You are an assistant answering questions ONLY using the provided Ayurveda source excerpts.

Rules:
1. Base your answer strictly on the excerpts given below - do not use outside or general knowledge, even if you know the answer independently.
2. Cite the source and page number for every claim, like: (Source: {source}, page {page}).
3. If the excerpts do not contain enough information to answer, say clearly that the corpus does not contain sufficient evidence to answer this question - do NOT guess, and do NOT fall back on general knowledge to fill the gap.
4. If the question is unrelated to the traditional medicine content covered by the corpus (e.g. unrelated technology, history, or science questions), state plainly that this is outside the scope of the corpus - do not attempt to answer it anyway.
5. If the question is about a DIFFERENT medical tradition (e.g. Traditional Chinese Medicine, Unani) that the corpus does not cover, say so explicitly rather than answering from general knowledge about that other tradition.
6. Respond in the SAME language the user asked the question in, where reasonably possible - the source excerpts may be in English even when the question is in Sanskrit, Hindi, or Arabic; translate your answer accordingly, but keep citations (source names, page numbers) as given.
7. This is educational/informational content about traditional medicine texts, not medical advice. Do not present dosages or treatment steps as direct instructions to follow - note that a qualified practitioner should be consulted before acting on anything discussed.
"""


def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source')}, page {c['metadata'].get('page_number')}]\n{c['content']}"
        for c in chunks)


def answer_question(question, top_k=5, model=MODEL_NAME, min_relevance=MIN_RELEVANCE_SCORE,
                    keyword_weight=KEYWORD_WEIGHT, use_llm_translation=False):
    results = multi_query_hybrid_search(question, top_k=top_k, keyword_weight=keyword_weight,
                                        use_llm=use_llm_translation, model=model)
    scores = [r["score"] for r in results]
    if not results or results[0]["score"] < min_relevance:
        return {"answer": "The corpus does not contain sufficient evidence to answer this question.",
                "sources": [], "retrieval_scores": scores, "model": model}

    user_message = (f"Source excerpts:\n\n{build_context(results)}\n\nQuestion: {question}\n\n"
                    "Answer using only the excerpts above, with citations for every claim.")
    text = llm_providers.generate(model, SYSTEM_PROMPT, user_message, max_tokens=1000)
    return {
        "answer": text,
        "model": model,
        "sources": [{"source": r["metadata"].get("source"), "page": r["metadata"].get("page_number"),
                     "category": r["metadata"].get("category"), "score": r["score"],
                     "snippet": r["content"][:280]} for r in results],
        "retrieval_scores": scores,
    }


TRANSLATE_DISCLAIMER = ("Automated translation - accuracy has not been verified, especially for "
                        "medical, botanical or technical terms. Treat with caution; when in doubt, "
                        "refer back to the original-language answer above.")

TRANSLATE_SYSTEM_PROMPT = """You are a translation assistant. Translate the given text accurately
into the requested language, preserving its meaning as closely as possible. Keep any citations
(source names, page numbers in parentheses) EXACTLY as given - do not translate them. Output ONLY
the translation itself, with no preamble, explanation, or notes before or after it."""


def translate_text(text, target_language, model=MODEL_NAME):
    """
    Translates `text` (typically an already-generated English answer) into
    `target_language` (e.g. "Sanskrit", "Arabic", "Chinese"). This is NOT
    fact-checked or verified for translation accuracy - the disclaimer is
    returned alongside the translation precisely so the caller surfaces it
    to the user rather than presenting the translation as authoritative.
    """
    user_message = f"Translate the following text into {target_language}:\n\n{text}"
    translation = llm_providers.generate(model, TRANSLATE_SYSTEM_PROMPT, user_message, max_tokens=1200)
    return {
        "translation": translation,
        "target_language": target_language,
        "disclaimer": TRANSLATE_DISCLAIMER,
    }


if __name__ == "__main__":
    res = answer_question("What is Ajirna and how is it treated?")
    print(res["answer"])
    for s in res["sources"]:
        print(f"  {s['source']} (page {s['page']}) - {s['score']:.3f}")

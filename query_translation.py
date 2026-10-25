"""Query expansion for the English / Sanskrit-transliteration mismatch.

The problem: a user searches in plain English ("cold shoulder pain"), but
the corpus is full of Ayurvedic terms written as Sanskrit transliterated
into English ("Amavata", "Vata prakopa", "Shita anga"...). Pure semantic
search helps some, but hybrid_search's keyword-overlap boost (and even the
embedding itself) can miss these because there's no shared vocabulary
between the query and the chunk.

Fix: query expansion / multi-query retrieval. For every search, also build
a "Sanskrit-leaning" variant of the query, run hybrid_search on BOTH the
original and the variant, then merge results by taking the best score each
chunk got under either phrasing.

Two layers, used together:
  1. GLOSSARY  - a curated English -> Sanskrit-transliteration lookup for
     common Ayurvedic vocabulary (doshas, body parts, symptoms, treatments).
     Free, instant, no network call, 100% predictable.
  2. llm_translate_query() - optional. For terms the glossary doesn't cover,
     asks an LLM (reusing llm_providers.generate(), same as the rest of the
     app) for the likely Sanskrit/Ayurvedic terms. Fails closed: any error
     just means that variant is skipped, search still works.

Usage (drop-in replacement for a plain hybrid_search call):
    from query_translation import multi_query_hybrid_search
    results = multi_query_hybrid_search(question, top_k=5, keyword_weight=0.2)
"""
import re
from embed_corpus import hybrid_search

# ---------------------------------------------------------------------------
# Layer 1: curated glossary. Keys are lowercase English phrases/words, values
# are the Sanskrit (IAST-ish, Latin-script) term(s) an Ayurvedic text is
# likely to actually use. Extend this as you find more gaps in real search
# results - it's the cheapest, most reliable lever you have.
# ---------------------------------------------------------------------------
GLOSSARY = {
    # --- doshas / constitution ---
    "wind": "vata", "air element": "vata", "bile": "pitta", "fire element": "pitta",
    "phlegm": "kapha", "mucus": "kapha", "water element": "kapha",
    "body type": "prakriti", "constitution": "prakriti", "imbalance": "vikriti dosha",
    "life force": "prana", "vital energy": "ojas",

    # --- digestion ---
    "digestive fire": "agni", "indigestion": "ajirna", "poor digestion": "manda agni",
    "toxins": "ama", "bloating": "adhmana", "constipation": "vibandha",
    "acidity": "amlapitta", "appetite loss": "arochaka",

    # --- common symptoms / conditions ---
    "cold shoulder": "amavata skandha", "frozen shoulder": "apabahuka",
  "joint pain": "sandhivata", "arthritis": "amavata", "back pain": "kati shoola",
    "lower back pain": "kati graha", "headache": "shirahshoola", "migraine": "ardhavabhedaka",
    "fever": "jvara", "cough": "kasa", "cold": "pratishyaya", "common cold": "pratishyaya",
    "asthma": "shwasa", "diabetes": "madhumeha", "obesity": "sthaulya",
    "skin disease": "kushtha", "hair loss": "khalitya", "insomnia": "anidra",
    "anxiety": "chittodvega", "fatigue": "klama", "weakness": "daurbalya",
    "piles": "arsha", "hemorrhoids": "arsha", "anemia": "pandu",

    # --- body / anatomy ---
    "shoulder": "skandha", "knee": "janu", "stomach": "udara", "heart": "hridaya",
    "liver": "yakrit", "lungs": "phuphusa", "head": "shiras", "throat": "kantha",

    # --- treatment / procedure ---
    "detox": "panchakarma", "oil massage": "abhyanga", "purging therapy": "virechana",
    "enema therapy": "basti", "nasal therapy": "nasya", "sweating therapy": "swedana",
    "diet": "pathya", "rejuvenation": "rasayana", "aphrodisiac therapy": "vajikarana",

    # --- common herbs (English name -> Sanskrit/Ayurvedic name) ---
    "ashwagandha": "ashwagandha", "winter cherry": "ashwagandha",
    "indian ginseng": "ashwagandha", "holy basil": "tulsi", "turmeric": "haridra",
    "licorice": "yashtimadhu", "gooseberry": "amalaki", "indian gooseberry": "amalaki",
    "neem": "nimba", "ginger": "shunthi", "fenugreek": "methi",
    "triphala": "triphala", "brahmi": "brahmi",
}

# Sort once, longest phrase first, so "cold shoulder" matches before "cold".
_GLOSSARY_KEYS = sorted(GLOSSARY.keys(), key=len, reverse=True)


def glossary_expand(query):
    """Returns a Sanskrit-leaning variant of `query`, or None if nothing in
    the glossary matched (no point searching twice with identical text)."""
    q_lower = query.lower()
    matched_terms = []
    covered = [False] * len(q_lower)

    for phrase in _GLOSSARY_KEYS:
        for m in re.finditer(re.escape(phrase), q_lower):
            start, end = m.start(), m.end()
            if any(covered[start:end]):
                continue  # don't double-match overlapping phrases
            matched_terms.append(GLOSSARY[phrase])
            for i in range(start, end):
                covered[i] = True

    if not matched_terms:
        return None
    # Append rather than replace: keeps the variant a valid, searchable
    # string even if the glossary term alone would be too sparse.
    return query + " " + " ".join(dict.fromkeys(matched_terms))  # dedup, keep order


# ---------------------------------------------------------------------------
# Layer 2: optional LLM-based translation for terms the glossary misses.
# Reuses the app's existing llm_providers module, so it respects whichever
# API keys/models are already configured - no new dependency.
# ---------------------------------------------------------------------------
_TRANSLATE_SYSTEM_PROMPT = (
    "You convert a plain-English health/Ayurveda query into the Sanskrit "
    "terminology (written in Latin script, not Devanagari) that a classical "
    "Ayurvedic text would use for the same concept. Reply with ONLY the "
    "Sanskrit terms, space-separated, no explanation, no punctuation. If you "
    "are not reasonably confident, reply with exactly: NONE"
)


def llm_translate_query(query, model=None):
    """Best-effort LLM translation. Returns a variant string, or None if the
    LLM is unavailable, errors, or isn't confident. Never raises."""
    try:
        import llm_providers
        model = model or llm_providers.DEFAULT_MODEL
        text = llm_providers.generate(model, _TRANSLATE_SYSTEM_PROMPT, query, max_tokens=60)
        text = (text or "").strip()
        if not text or text.upper() == "NONE" or len(text) > 200:
            return None
        return query + " " + text
    except Exception:
        return None


def get_query_variants(query, use_llm=False, model=None):
    """Returns a deduplicated list of query strings to search with:
    [original, glossary variant (if any), llm variant (if requested/any)]."""
    variants = [query]
    g = glossary_expand(query)
    if g and g != query:
        variants.append(g)
    if use_llm:
        l = llm_translate_query(query, model=model)
        if l and l not in variants:
            variants.append(l)
    return variants


def multi_query_hybrid_search(query, top_k=5, candidate_pool=25, keyword_weight=0.3,
                              use_llm=False, model=None, **kwargs):
    """Drop-in replacement for embed_corpus.hybrid_search(). Runs retrieval
    for the original query AND its Sanskrit-leaning variant(s), merges by
    chunk id taking the best score seen under any phrasing, and returns the
    top_k merged results - same shape as hybrid_search's return value, plus
    a "matched_variant" field showing which phrasing won each chunk."""
    variants = get_query_variants(query, use_llm=use_llm, model=model)

    best_by_id = {}
    for variant in variants:
        for r in hybrid_search(variant, top_k=candidate_pool,
                               candidate_pool=candidate_pool,
                               keyword_weight=keyword_weight, **kwargs):
            prev = best_by_id.get(r["id"])
            if prev is None or r["score"] > prev["score"]:
                r = dict(r)
                r["matched_variant"] = variant
                best_by_id[r["id"]] = r

    merged = sorted(best_by_id.values(), key=lambda r: -r["score"])[:top_k]
    return merged


if __name__ == "__main__":
    # Quick offline check of the glossary layer (no corpus/model needed).
    tests = ["cold shoulder pain", "what helps with indigestion?", "treatment for joint pain",
             "ashwagandha benefits", "how to improve digestive fire"]
    for q in tests:
        print(f"{q!r}\n  -> {get_query_variants(q)}\n")

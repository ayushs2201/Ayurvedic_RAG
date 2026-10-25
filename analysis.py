"""Corpus analysis (from your report notebook) returning JSON for the website.
Edit compute() to add new statistics - anything you return appears in
/api/analysis, and the website's Analysis section renders the fields below."""
import re, statistics
from collections import Counter
import numpy as np

STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was were be been being this that
these those it its it's he she they them his her their there here which who whom what when where why how not no nor so than
then too very can will just should now also into about over under after before between during while above below up down
out off again further i you we me my your our us""".split())
DEV = (0x0900, 0x097F)


def _script_class(text):
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "latin-only"
    r = sum(1 for c in alpha if DEV[0] <= ord(c) <= DEV[1]) / len(alpha)
    return "devanagari-heavy" if r >= 0.3 else "mixed" if r > 0.02 else "latin-only"


def _hist(values, bins, rng=None):
    if not values:
        return {"edges": [], "counts": []}
    c, e = np.histogram(values, bins=bins, range=rng)
    return {"edges": [round(float(x), 1) for x in e], "counts": [int(x) for x in c]}


def compute(corpus):
    sources, cats, scripts, freq = Counter(), Counter(), Counter(), Counter()
    words, ocr_conf, low = [], [], []
    ocr_used = est_pages = 0
    for ch in corpus:
        meta, text = ch.get("metadata", {}), ch.get("content", "")
        words.append(len(text.split()))
        sources[meta.get("source", "unknown")] += 1
        cats[meta.get("category", "unknown")] += 1
        scripts[_script_class(text)] += 1
        freq.update(w for w in re.findall(r"[A-Za-z']{3,}", text.lower()) if w not in STOPWORDS)
        if meta.get("page_ocr_used"):
            ocr_used += 1
            conf = meta.get("page_ocr_confidence")
            if conf is not None:
                ocr_conf.append(conf)
                if conf < 70:
                    low.append({"id": ch.get("id"), "source": meta.get("source"), "page": meta.get("page_number"),
                                "confidence": round(conf, 1), "snippet": text[:150].replace("\n", " ")})
        if meta.get("page_number_estimated"):
            est_pages += 1
    p99 = float(np.percentile(words, 99)) if words else 0
    return {
        "total_chunks": len(corpus), "unique_sources": len(sources), "unique_categories": len(cats),
        "total_words": int(sum(words)), "avg_words": round(statistics.mean(words), 1) if words else 0,
        "median_words": statistics.median(words) if words else 0,
        "ocr_used": ocr_used, "estimated_pages": est_pages, "real_pages": len(corpus) - est_pages,
        "categories": cats.most_common(15), "sources": sources.most_common(15),
        "length_hist": _hist(words, 30, (0, p99)), "scripts": dict(scripts),
        "ocr_hist": _hist(ocr_conf, 20, (0, 100)), "top_words": freq.most_common(25),
        "low_conf": sorted(low, key=lambda x: x["confidence"])[:25],
    }

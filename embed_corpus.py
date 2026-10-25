"""Embedding + retrieval (your notebook code, packaged as a module).

Build the index once:   python embed_corpus.py
The website then calls hybrid_search() with the SAME SentenceTransformer.
"""
import json, os, re, threading
import numpy as np
from sentence_transformers import SentenceTransformer
import config

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

DEFAULT_MODEL_NAME = config.EMBED_MODEL


def load_corpus(json_path=config.CORPUS_JSON_PATH):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(corpus=None, model_name=DEFAULT_MODEL_NAME, batch_size=32):
    if corpus is None:
        corpus = load_corpus()
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [c["content"] for c in corpus]
    print(f"Embedding {len(texts)} chunks...")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    np.save(config.EMBEDDINGS_PATH, emb)
    with open(config.INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb.astype("float32"))
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        print("Saved FAISS index")
    print(f"Saved {len(emb)} embeddings")
    return emb, corpus, model


_lock = threading.Lock()
_cache = {"model": None, "emb": None, "corpus": None, "faiss": None}


def _load_cached(model_name=DEFAULT_MODEL_NAME):
    """Loads model, embeddings, corpus and FAISS index once, then reuses them."""
    with _lock:
        if _cache["model"] is None:
            _cache["model"] = SentenceTransformer(model_name)
        if _cache["emb"] is None:
            _cache["emb"] = np.load(config.EMBEDDINGS_PATH)
        if _cache["corpus"] is None:
            with open(config.INDEX_METADATA_PATH, "r", encoding="utf-8") as f:
                _cache["corpus"] = json.load(f)
        if FAISS_AVAILABLE and _cache["faiss"] is None and os.path.exists(config.FAISS_INDEX_PATH):
            _cache["faiss"] = faiss.read_index(config.FAISS_INDEX_PATH)
    return _cache["model"], _cache["emb"], _cache["corpus"], _cache["faiss"]


def _dense(query_vec, k, emb, faiss_index):
    if faiss_index is not None:
        s, i = faiss_index.search(query_vec.astype("float32"), k)
        return s[0], i[0]
    sims = emb @ query_vec[0]
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx


def search(query, top_k=5, model_name=DEFAULT_MODEL_NAME):
    model, emb, corpus, fi = _load_cached(model_name)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = _dense(qv, top_k, emb, fi)
    return [{"score": float(s), "id": corpus[i]["id"], "content": corpus[i]["content"],
             "metadata": corpus[i]["metadata"]} for s, i in zip(scores, idx)]


def _keyword_terms(text):
    return set(re.findall(r"[A-Za-z']{3,}", text.lower()))


def hybrid_search(query, top_k=5, candidate_pool=25, keyword_weight=0.3, model_name=DEFAULT_MODEL_NAME):
    model, emb, corpus, fi = _load_cached(model_name)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    dense_scores, indices = _dense(qv, candidate_pool, emb, fi)
    qt = _keyword_terms(query)
    ranked = []
    for d, i in zip(dense_scores, indices):
        ch = corpus[i]
        terms = _keyword_terms(ch["content"]) | _keyword_terms(ch["metadata"].get("source", ""))
        kw = len(qt & terms) / max(len(qt), 1)
        ranked.append(((1 - keyword_weight) * float(d) + keyword_weight * kw, float(d), kw, i))
    ranked.sort(key=lambda x: -x[0])
    return [{"score": f, "dense_score": d, "keyword_score": k, "id": corpus[i]["id"],
             "content": corpus[i]["content"], "metadata": corpus[i]["metadata"]}
            for f, d, k, i in ranked[:top_k]]


if __name__ == "__main__":
    build_index()
    for r in search("what is the treatment for indigestion?", top_k=3):
        print(f"[{r['score']:.3f}] {r['id']}\n  {r['content'][:200]}")

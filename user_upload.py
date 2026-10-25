"""Lets a user upload their own .txt or .pdf file, builds a small in-memory
index for THAT file only (kept separate from the main corpus - uploads never
get merged into the shared knowledge base), and answers questions scoped to
just that file. Also reused for the per-upload NLP analysis (via analysis.py,
unchanged - it already just takes any list of chunks).

Design choice: uploads stay session-scoped (in-memory, not written to disk or
to the main corpus/embeddings). This avoids one user's upload polluting or
poisoning the shared corpus everyone else queries, and avoids needing an
auth/ownership system for a prototype. Sessions are cleared after SESSION_TTL
seconds of inactivity so memory doesn't grow unbounded on a long-running server.
"""
import io
import os
import re
import time
import threading
import uuid

import numpy as np

import config
import embed_corpus
import llm_providers

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB - generous for a single uploaded doc/article
SESSION_TTL = 2 * 60 * 60  # 2 hours of inactivity before an upload's index is dropped
MIN_WORDS_PER_CHUNK = 30
MAX_WORDS_PER_CHUNK = 500
OVERLAP_WORDS = 20

SYSTEM_PROMPT_UPLOAD = """You are an assistant answering questions ONLY using the excerpts
provided below, which come from a document the user uploaded themselves (NOT the main
Ayurveda text corpus).

Rules:
1. Base your answer strictly on the excerpts given - do not use outside or general knowledge.
2. Cite the page number for every claim, like: (page {page}).
3. If the excerpts do not contain enough information to answer, say clearly that the
   uploaded document does not contain sufficient evidence to answer this question.
4. This is educational/informational content, not medical advice. Do not present dosages
   or treatment steps as direct instructions - note that a qualified practitioner should
   be consulted before acting on anything discussed, especially since this document has
   not been reviewed or verified by the project's own curated corpus.
"""


class UploadError(Exception):
    pass


_sessions = {}
_lock = threading.Lock()


def _cleanup_expired():
    cutoff = time.time() - SESSION_TTL
    expired = [uid for uid, s in _sessions.items() if s["last_used"] < cutoff]
    for uid in expired:
        del _sessions[uid]


# ---------- extraction ----------

def _read_text_file(file_bytes):
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UploadError("Could not decode this file as text.")


def _extract_pdf_pages(file_bytes):
    """Returns list of (page_num, page_text). Native text extraction only -
    no OCR in this lean upload path, so a scanned/image-only PDF will come
    back with little or no text; callers should surface that clearly rather
    than silently producing an empty, unusable index."""
    if not PYMUPDF_AVAILABLE:
        raise UploadError("PDF support isn't installed on the server (pip install pymupdf).")
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise UploadError(f"Could not open this PDF: {e}")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text() or ""
        if text.strip():
            pages.append((i, text))
    doc.close()
    return pages


def extract_text(file_bytes, filename):
    """Returns list of (page_num, page_text). .txt files are treated as one page."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return [(1, _read_text_file(file_bytes))]
    elif ext == ".pdf":
        pages = _extract_pdf_pages(file_bytes)
        if not pages:
            raise UploadError(
                "No extractable text found in this PDF - it may be a scanned image "
                "without a text layer, which this upload path does not OCR."
            )
        return pages
    else:
        raise UploadError("Only .txt and .pdf files are supported.")


# ---------- chunking (paragraph-aware, same approach as the main corpus pipeline) ----------

def _chunk_page(page_text, page_num, source_name, chunk_idx_start):
    paragraphs = [p for p in page_text.split("\n\n") if p.strip()]
    chunks = []
    chunk_idx = chunk_idx_start
    buffer, buffer_word_count = [], 0

    def flush():
        nonlocal buffer, buffer_word_count, chunk_idx
        if not buffer:
            return
        combined = "\n\n".join(buffer).strip()
        if len(combined) > 10:
            chunks.append({
                "id": f"{source_name}-chunk-{chunk_idx}",
                "content": combined,
                "metadata": {"source": source_name, "category": "upload", "page_number": page_num},
            })
            chunk_idx += 1
        buffer, buffer_word_count = [], 0

    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if len(words) > MAX_WORDS_PER_CHUNK:
            flush()
            step = max(1, MAX_WORDS_PER_CHUNK - OVERLAP_WORDS)
            for i in range(0, len(words), step):
                sub = " ".join(words[i:i + MAX_WORDS_PER_CHUNK]).strip()
                if len(sub) > 10:
                    chunks.append({
                        "id": f"{source_name}-chunk-{chunk_idx}",
                        "content": sub,
                        "metadata": {"source": source_name, "category": "upload", "page_number": page_num},
                    })
                    chunk_idx += 1
                if i + MAX_WORDS_PER_CHUNK >= len(words):
                    break
            continue
        buffer.append(para)
        buffer_word_count += len(words)
        if buffer_word_count >= MIN_WORDS_PER_CHUNK:
            flush()
    flush()
    return chunks, chunk_idx


def chunk_document(pages, source_name):
    all_chunks = []
    chunk_idx = 0
    for page_num, page_text in pages:
        page_chunks, chunk_idx = _chunk_page(page_text, page_num, source_name, chunk_idx)
        all_chunks.extend(page_chunks)
    return all_chunks


# ---------- build + search a per-upload index ----------

def build_upload_index(file_bytes, filename):
    if len(file_bytes) > MAX_FILE_BYTES:
        raise UploadError(f"File is too large (max {MAX_FILE_BYTES // (1024 * 1024)} MB).")

    source_name = os.path.splitext(os.path.basename(filename))[0]
    pages = extract_text(file_bytes, filename)
    chunks = chunk_document(pages, source_name)
    if not chunks:
        raise UploadError("Could not extract any usable text from this file.")

    # Reuse the SAME already-loaded sentence-transformer as the main corpus -
    # no second model load, and guarantees compatible embeddings.
    model = embed_corpus._load_cached()[0]
    texts = [c["content"] for c in chunks]
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    upload_id = uuid.uuid4().hex[:12]
    with _lock:
        _cleanup_expired()
        _sessions[upload_id] = {
            "chunks": chunks, "emb": emb, "filename": filename,
            "pages": len(pages), "last_used": time.time(),
        }
    return upload_id, chunks


def get_session(upload_id):
    with _lock:
        s = _sessions.get(upload_id)
        if s is None:
            raise UploadError("This upload has expired or was not found - upload the file again.")
        s["last_used"] = time.time()
        return s


def search_upload(upload_id, query, top_k=5):
    s = get_session(upload_id)
    model = embed_corpus._load_cached()[0]
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = s["emb"] @ qv
    idx = np.argsort(-sims)[:top_k]
    return [{"score": float(sims[i]), "id": s["chunks"][i]["id"],
             "content": s["chunks"][i]["content"], "metadata": s["chunks"][i]["metadata"]}
            for i in idx]


def _build_context(chunks):
    return "\n\n---\n\n".join(f"[page {c['metadata'].get('page_number')}]\n{c['content']}" for c in chunks)


def answer_from_upload(upload_id, question, model=None, top_k=5, min_relevance=0.3):
    model = model or llm_providers.DEFAULT_MODEL
    results = search_upload(upload_id, question, top_k=top_k)
    scores = [r["score"] for r in results]
    if not results or results[0]["score"] < min_relevance:
        return {"answer": "The uploaded document does not contain sufficient evidence to answer this question.",
                "sources": [], "retrieval_scores": scores, "model": model}

    user_message = (f"Excerpts from the uploaded document:\n\n{_build_context(results)}\n\n"
                     f"Question: {question}\n\nAnswer using only the excerpts above, with page citations.")
    text = llm_providers.generate(model, SYSTEM_PROMPT_UPLOAD, user_message, max_tokens=1000)
    return {
        "answer": text, "model": model,
        "sources": [{"source": r["metadata"].get("source"), "page": r["metadata"].get("page_number"),
                     "category": "upload", "score": r["score"], "snippet": r["content"][:280]} for r in results],
        "retrieval_scores": scores,
    }

"""Ayurveda RAG website - run:  python app.py   then open http://localhost:5000"""
import json, os, random, threading, time, uuid
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
import config, llm_providers, analysis, image_search, speech
import embed_corpus
import user_upload
from generate_answer import answer_question, translate_text

app = Flask(__name__, static_folder="static", static_url_path="")
STATE = {"ready": False, "error": None, "chunks": 0}
_an_lock, _res_lock = threading.Lock(), threading.Lock()


def warmup():
    """Load the sentence-transformer, embeddings, FAISS index and corpus once."""
    try:
        _, _, corpus, _ = embed_corpus._load_cached()
        STATE.update(chunks=len(corpus), ready=True)
        print(f"Ready: {len(corpus):,} chunks loaded.")
        threading.Thread(target=speech.warm, daemon=True).start()  # preload Whisper in the background
    except Exception as e:  # missing files etc.
        STATE["error"] = f"{type(e).__name__}: {e}"
        print("Startup error:", STATE["error"])


threading.Thread(target=warmup, daemon=True).start()


def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    return jsonify({**STATE, "voice": speech.capabilities()})


@app.post("/api/stt")
def stt():
    f = request.files.get("audio")
    if not f:
        return err("No recording received.")
    try:
        return jsonify({"text": speech.transcribe(f.read(), request.form.get("lang", "en-US"))})
    except speech.SpeechError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(f"Transcription failed: {e}", 500)


@app.post("/api/tts")
def tts():
    d = request.get_json(force=True, silent=True) or {}
    try:
        audio, mime = speech.synthesize(d.get("text", ""), d.get("lang", "en-US"))
        return Response(audio, mimetype=mime)
    except speech.SpeechError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(f"Text to speech failed: {e}", 500)


@app.get("/api/models")
def models():
    return jsonify({"default": llm_providers.DEFAULT_MODEL, "models": llm_providers.list_models()})


@app.post("/api/chat")
def chat():
    d = request.get_json(force=True, silent=True) or {}
    q = (d.get("question") or "").strip()
    if not q:
        return err("Type a question first.")
    if not STATE["ready"]:
        return err(STATE["error"] or "The search model is still loading. Try again in a moment.", 503)
    model = d.get("model") or llm_providers.DEFAULT_MODEL
    top_k = max(1, min(int(d.get("top_k") or 5), 10))
    herb = image_search.extract_image_request(q)
    images = []
    if herb:
        try:
            images = image_search.fetch_reference_photos(herb, 4)
        except Exception:
            images = []
    try:
        result = answer_question(q, top_k=top_k, model=model)
    except llm_providers.ProviderError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(f"Could not generate an answer: {e}", 500)
    result.update(images=images, image_query=herb)
    return jsonify(result)


@app.get("/api/images")
def images():
    herb = (request.args.get("herb") or "").strip()
    if not herb:
        return err("Enter a herb name.")
    try:
        return jsonify({"herb": herb, "images": image_search.fetch_reference_photos(herb, 6)})
    except Exception as e:
        return err(f"Image search failed (check your internet connection): {e}", 502)


@app.post("/api/identify")
def identify():
    f = request.files.get("image")
    if not f:
        return err("Choose a photo first.")
    if f.mimetype not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        return err("Use a JPG, PNG, WebP or GIF image.")
    model = request.form.get("model") or llm_providers.DEFAULT_MODEL
    try:
        if llm_providers.get_model(model)["provider"] != "anthropic":
            model = llm_providers.DEFAULT_MODEL  # photo analysis needs a vision model
        text = image_search.identify_from_user_photo(f.read(), f.mimetype, model)
        return jsonify({"description": text, "model": model})
    except Exception as e:
        return err(f"Could not analyse the photo: {e}", 500)


@app.get("/api/analysis")
def get_analysis():
    refresh = request.args.get("refresh") == "1"
    if not STATE["ready"]:
        return err(STATE["error"] or "Still loading.", 503)
    try:
        mtime = os.path.getmtime(config.INDEX_METADATA_PATH)
    except OSError:
        mtime = 0
    with _an_lock:
        if not refresh and os.path.exists(config.ANALYSIS_CACHE):
            with open(config.ANALYSIS_CACHE, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("corpus_mtime") == mtime:
                return jsonify(cached)
        corpus = embed_corpus._load_cached()[2]
        data = analysis.compute(corpus)
        data.update(corpus_mtime=mtime, generated=time.strftime("%Y-%m-%d %H:%M"))
        with open(config.ANALYSIS_CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return jsonify(data)


@app.get("/api/dataset")
def dataset():
    if not STATE["ready"]:
        return err("Still loading.", 503)
    corpus = embed_corpus._load_cached()[2]
    q = (request.args.get("q") or "").lower().strip()
    if q:
        hits = [c for c in corpus if q in c["content"].lower() or q in c["metadata"].get("source", "").lower()]
        total = len(hits)
        rows = hits[:15]
    else:
        total, rows = len(corpus), random.sample(corpus, min(15, len(corpus)))
    return jsonify({"total": total, "rows": [
        {"id": c["id"], "source": c["metadata"].get("source"), "category": c["metadata"].get("category"),
         "page": c["metadata"].get("page_number"), "text": c["content"][:320]} for c in rows]})


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("Choose a .txt or .pdf file first.")
    if not STATE["ready"]:
        return err(STATE["error"] or "The search model is still loading. Try again in a moment.", 503)
    try:
        upload_id, chunks = user_upload.build_upload_index(f.read(), f.filename)
    except user_upload.UploadError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(f"Could not process this file: {e}", 500)
    return jsonify({
        "upload_id": upload_id, "filename": f.filename, "chunk_count": len(chunks),
        "pages": user_upload.get_session(upload_id)["pages"],
        "analysis": analysis.compute(chunks),
    })


@app.post("/api/ask_uploaded")
def ask_uploaded():
    d = request.get_json(force=True, silent=True) or {}
    upload_id = (d.get("upload_id") or "").strip()
    q = (d.get("question") or "").strip()
    if not upload_id:
        return err("Upload a file first.")
    if not q:
        return err("Type a question first.")
    model = d.get("model") or llm_providers.DEFAULT_MODEL
    top_k = max(1, min(int(d.get("top_k") or 5), 10))
    try:
        result = user_upload.answer_from_upload(upload_id, q, model=model, top_k=top_k)
    except user_upload.UploadError as e:
        return err(str(e), 404)
    except llm_providers.ProviderError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(f"Could not generate an answer: {e}", 500)
    return jsonify(result)


@app.post("/api/translate")
def translate():
    d = request.get_json(force=True, silent=True) or {}
    text = (d.get("text") or "").strip()
    target_language = (d.get("target_language") or "").strip()
    if not text:
        return err("Nothing to translate.")
    if not target_language:
        return err("Choose a target language.")
    model = d.get("model") or llm_providers.DEFAULT_MODEL
    try:
        return jsonify(translate_text(text, target_language, model=model))
    except llm_providers.ProviderError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(f"Translation failed: {e}", 500)


@app.get("/report")
def report():
    if os.path.exists(config.REPORT_PATH):
        return send_file(config.REPORT_PATH)
    return "Report file not found. Set REPORT_PATH in config.py.", 404


def _read_results():
    if os.path.exists(config.RESULTS_PATH):
        with open(config.RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/results")
def results_list():
    return jsonify(_read_results())


@app.post("/api/results")
def results_add():
    d = request.get_json(force=True, silent=True) or {}
    try:
        row = {"id": uuid.uuid4().hex[:8], "name": str(d["name"])[:80], "model": str(d["model"])[:60],
               "recall": float(d["recall"]), "accuracy": float(d["accuracy"]),
               "faithfulness": float(d["faithfulness"]), "latency": float(d["latency"]),
               "notes": str(d.get("notes", ""))[:300], "date": time.strftime("%Y-%m-%d")}
    except (KeyError, ValueError):
        return err("Fill in every metric with a number.")
    with _res_lock:
        rows = _read_results()
        rows.append(row)
        with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return jsonify(row)


@app.delete("/api/results/<rid>")
def results_delete(rid):
    with _res_lock:
        rows = [r for r in _read_results() if r["id"] != rid]
        with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return jsonify({"ok": True})


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

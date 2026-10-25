# Ayurveda RAG website

White and green website for your RAG project. It runs entirely on your machine.

## What it does
| Feature | Where |
|---|---|
| Chatbot with cited answers (LaBSE + FAISS + hybrid search) | `embed_corpus.py`, `generate_answer.py` |
| Choose the LLM from a dropdown | `llm_providers.py` (`MODELS` list) |
| Speech to text (Whisper) and text to speech (edge-tts / pyttsx3) | `speech.py` |
| Herb photo search and photo description | `image_search.py` |
| Dataset browser and corpus analysis | `analysis.py`, `/api/dataset`, `/api/analysis` |
| Published results table | `/api/results`, stored in `data/results.json` |

## 1. Install
```
python -m venv venv
venv\Scripts\activate            (Windows)   or   source venv/bin/activate
pip install -r requirements.txt
```
Your GPU needs the CUDA build of PyTorch you already installed (cu128). Use `faiss-cpu`, since `faiss-gpu` has no pip wheel. If faiss will not install, the code falls back to numpy search.

## 2. Point it at your data
Put these files in the `data/` folder, or point to the folder where they already are:
```
set AYUR_DATA_DIR=D:\RMIT\SEM 2\Case Studies in Data Science\Ayurvedic_RAG
```
Needed: `ayurveda_corpus_clean.json`, `ayurveda_embeddings.npy`, `ayurveda_index_metadata.json`, `ayurveda_faiss.index`.
Optional: `ayurveda_corpus_report.html` (opened by the "Open full report" button).

If the embeddings do not exist yet, build them once: `python embed_corpus.py`.

## 3. Choose your models
```
set ANTHROPIC_API_KEY=sk-ant-...      (Claude models)
set OPENAI_API_KEY=sk-...             (optional, GPT-4o)
set GROQ_API_KEY=gsk_...              (FREE - Llama 3.3 70B / 3.1 8B. Get a key at https://console.groq.com)
```
Free local models: install Ollama, run `ollama pull qwen2.5:7b`, and it appears in the dropdown automatically.
Models without a key or install show as "unavailable".

**If you don't want to pay for Anthropic/OpenAI at all**, Groq is the easiest free option: sign up, create a key, set `GROQ_API_KEY`, and "Llama 3.3 70B (Groq, free)" appears in the model dropdown with no other setup - runs in the cloud, no local GPU needed. Ollama is free too but only works while Ollama is running on the same machine as the server, so it's better suited to local development than a deployed site.

To add a model, add one line to `MODELS` in `llm_providers.py`:
```python
{"id": "claude-opus-5-5", "label": "My label", "provider": "anthropic"},
```

## 4. Run
```
python app.py
```
Open http://localhost:5000. The first start takes a minute while LaBSE and the index load; the page shows a loading banner.

## Notes
- **Voice runs in Python** (`speech.py`). The browser only records your microphone and plays the reply.
  - Listening: Whisper (`openai/whisper-small`) runs locally on your GPU through your existing PyTorch. It is downloaded from Hugging Face the first time. For better Hindi, set `WHISPER_MODEL=openai/whisper-medium`.
  - Speaking: `edge-tts` gives natural English (US and India) and Hindi voices but needs internet. If it is unavailable the server falls back to `pyttsx3`, which is fully offline and uses the voices installed in Windows. Force one with `TTS_ENGINE=edge` or `TTS_ENGINE=pyttsx3`.
  - The microphone needs permission on http://localhost (Chrome, Edge or Firefox all work, since recognition no longer depends on the browser).
- The loading screen and thinking indicators cycle through herb icons from `static/img/herbs/`. Add or replace SVG files there and list the names in the `HERBS` array in `static/index.html`.
- Animations respect the operating system's "reduce motion" setting.
- Herb photos and the photo description need internet access (Wikimedia and the Anthropic API).
- When your analysis changes, edit `compute()` in `analysis.py`, then press Recalculate on the website.
- `/api/results` has no login. That is fine on your own machine; add authentication before exposing the server publicly.
- Answers are educational and not medical advice; the system prompt in `generate_answer.py` already says so.

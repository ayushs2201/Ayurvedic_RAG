"""Central settings. Change paths here, or set the AYUR_DATA_DIR environment
variable to the folder that holds your corpus/embeddings, e.g. on Windows:
    set AYUR_DATA_DIR=D:\\RMIT\\SEM 2\\Case Studies in Data Science\\Ayurvedic_RAG
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"D:\RMIT\SEM 2\Case Studies in Data Science\Ayurvedic_RAG"

CORPUS_JSON_PATH    = os.path.join(DATA_DIR, "ayurveda_corpus_clean.json")
EMBEDDINGS_PATH     = os.path.join(DATA_DIR, "ayurveda_embeddings.npy")
INDEX_METADATA_PATH = os.path.join(DATA_DIR, "ayurveda_index_metadata.json")
FAISS_INDEX_PATH    = os.path.join(DATA_DIR, "ayurveda_faiss.index")

# Optional: your existing HTML report is served at /report if this file exists
REPORT_PATH    = os.path.join(DATA_DIR, "ayurveda_corpus_report.html")
ANALYSIS_CACHE = os.path.join(DATA_DIR, "analysis_cache.json")
RESULTS_PATH   = os.path.join(DATA_DIR, "results.json")

# Must be the SAME model used to build the index
EMBED_MODEL = "sentence-transformers/LaBSE"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ---------- Voice (Python side) ----------
# Speech to text: OpenAI Whisper running locally through Hugging Face transformers + your PyTorch/GPU.
# Options: openai/whisper-tiny, -base, -small (default), -medium (better for Hindi, needs more VRAM)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-small")

# Text to speech: "auto" tries edge-tts (natural voices, needs internet) then pyttsx3 (fully offline).
# Force one with "edge" or "pyttsx3".
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto")
EDGE_VOICES = {"en-US": "en-US-AriaNeural", "en-IN": "en-IN-NeerjaNeural", "hi-IN": "hi-IN-SwaraNeural"}

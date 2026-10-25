"""Speech in and out, entirely in Python.

  transcribe(wav_bytes, lang)  -> text     (Whisper, local, uses your GPU if PyTorch can)
  synthesize(text, lang)       -> (audio_bytes, mimetype)   (edge-tts, or pyttsx3 offline)

The browser only records the microphone and plays the returned audio; it does
the recognition and voice generation nowhere. The browser sends 16 kHz mono
WAV, so no ffmpeg install is needed."""
import asyncio, importlib.util, io, os, re, tempfile, threading, wave
import numpy as np
import config


class SpeechError(Exception):
    pass


_asr_lock, _tts_lock = threading.Lock(), threading.Lock()
_asr_pipe = None
LANG_NAMES = {"en": "english", "hi": "hindi"}


def capabilities():
    has = lambda m: importlib.util.find_spec(m) is not None
    return {"stt": has("transformers") and has("torch"), "stt_model": config.WHISPER_MODEL,
            "stt_loaded": _asr_pipe is not None,
            "tts": [e for e, m in (("edge", "edge_tts"), ("pyttsx3", "pyttsx3")) if has(m)]}


# ------------------------------------------------------------------ speech to text
def _get_asr():
    global _asr_pipe
    with _asr_lock:
        if _asr_pipe is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError:
                raise SpeechError("Speech input needs: pip install transformers torch")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper ({config.WHISPER_MODEL}) on {device}...")
            _asr_pipe = pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL, device=device)
    return _asr_pipe


def warm():
    try:
        _get_asr()
    except Exception as e:
        print("Whisper warm-up skipped:", e)


def _wav_to_array(data):
    try:
        with wave.open(io.BytesIO(data)) as w:
            ch, sr, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        raise SpeechError("The recording could not be read. Try again.")
    if width != 2:
        raise SpeechError("Unsupported audio format (expected 16-bit WAV).")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000 and len(a):
        a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)), np.arange(len(a)), a).astype(np.float32)
    return a


def transcribe(wav_bytes, lang="en-US"):
    audio = _wav_to_array(wav_bytes)
    if len(audio) < 3200:  # under 0.2 s
        raise SpeechError("That recording was too short. Hold on a little longer.")
    if float(np.abs(audio).max()) < 0.005:
        raise SpeechError("No sound was picked up. Check your microphone.")
    pipe = _get_asr()
    language = LANG_NAMES.get(lang.split("-")[0].lower(), "english")
    with _asr_lock:  # one transcription at a time
        out = pipe({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30,
                   generate_kwargs={"language": language, "task": "transcribe"})
    return (out.get("text") or "").strip()


# ------------------------------------------------------------------ text to speech
def clean_for_speech(text, limit=1400):
    t = re.sub(r"\((?:Source|Sources):[^)]*\)", "", text)          # spoken citations are noise
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[*_#`>]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    if len(t) > limit:
        cut = t[:limit]
        t = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
    return t


def _edge(text, lang):
    import edge_tts
    voice = config.EDGE_VOICES.get(lang, config.EDGE_VOICES["en-US"])

    async def run():
        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    data = asyncio.run(run())
    if not data:
        raise SpeechError("edge-tts returned no audio (check your internet connection).")
    return data, "audio/mpeg"


def _pyttsx3(text, lang):
    import pyttsx3
    with _tts_lock:
        try:
            import pythoncom  # Windows: SAPI needs COM in worker threads
            pythoncom.CoInitialize()
        except ImportError:
            pass
        engine = pyttsx3.init()
        want = "hindi" if lang.startswith("hi") else "english"
        for v in engine.getProperty("voices"):
            if want in (v.name or "").lower() or any(want[:2] in str(l).lower() for l in (v.languages or [])):
                engine.setProperty("voice", v.id)
                break
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            engine.stop()
            with open(path, "rb") as f:
                data = f.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    if not data:
        raise SpeechError("pyttsx3 produced no audio.")
    return data, "audio/wav"


def synthesize(text, lang="en-US"):
    text = clean_for_speech(text)
    if not text:
        raise SpeechError("There is nothing to read aloud.")
    order = {"edge": ["edge"], "pyttsx3": ["pyttsx3"]}.get(config.TTS_ENGINE, ["edge", "pyttsx3"])
    errors = []
    for eng in order:
        try:
            return (_edge if eng == "edge" else _pyttsx3)(text, lang)
        except ImportError:
            errors.append(f"{eng} is not installed")
        except SpeechError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{eng}: {e}")
    raise SpeechError("Text to speech failed: " + "; ".join(errors) + ". Install with: pip install edge-tts pyttsx3")

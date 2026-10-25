"""LLM model registry. To offer another model in the website's dropdown,
add ONE line to MODELS. The website reads this list from /api/models.

provider = "anthropic" (needs ANTHROPIC_API_KEY)
           "openai"    (needs OPENAI_API_KEY and `pip install openai`)
           "groq"      (FREE tier - needs GROQ_API_KEY and `pip install groq`.
                        Get a free key at https://console.groq.com)
           "ollama"    (free local models; install Ollama and `ollama pull <id>`)
"""
import os, time
import requests
import config

MODELS = [
    {"id": "claude-sonnet-5",           "label": "Claude Sonnet 5 (balanced)",  "provider": "anthropic"},
    {"id": "claude-opus-5-5",           "label": "Claude Opus 5.5 (deepest)",   "provider": "anthropic"},
    {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 (fastest)",  "provider": "anthropic"},
    {"id": "gpt-4o",                    "label": "GPT-4o",                       "provider": "openai"},
    {"id": "openai/gpt-oss-120b",       "label": "GPT-OSS 120B (Groq, free)",   "provider": "groq"},
    {"id": "openai/gpt-oss-20b",        "label": "GPT-OSS 20B (Groq, free, fastest)", "provider": "groq"},
    {"id": "qwen2.5:7b",                "label": "Qwen 2.5 7B (local)",          "provider": "ollama"},
]
DEFAULT_MODEL = "claude-sonnet-5"
GROQ_API_KEY=''


class ProviderError(Exception):
    pass


_ollama_cache = {"t": 0, "names": set()}


def _ollama_models():
    if time.time() - _ollama_cache["t"] > 10:
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=0.7)
            _ollama_cache["names"] = {m["name"] for m in r.json().get("models", [])}
        except Exception:
            _ollama_cache["names"] = set()
        _ollama_cache["t"] = time.time()
    return _ollama_cache["names"]


def _status(m):
    """Returns (available, hint)."""
    p = m["provider"]
    if p == "anthropic":
        return (bool(os.environ.get("ANTHROPIC_API_KEY")), "set ANTHROPIC_API_KEY")
    if p == "openai":
        return (bool(os.environ.get("OPENAI_API_KEY")), "set OPENAI_API_KEY")
    if p == "groq":
        return (bool(os.environ.get("GROQ_API_KEY")), "set GROQ_API_KEY (free at console.groq.com)")
    if p == "ollama":
        return (m["id"] in _ollama_models(), f"run: ollama pull {m['id']}")
    return (False, "unknown provider")


def list_models():
    out = []
    for m in MODELS:
        ok, hint = _status(m)
        out.append({**m, "available": ok, "hint": "" if ok else hint})
    return out


def get_model(model_id):
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise ProviderError(f"Unknown model '{model_id}'.")


def generate(model_id, system, user, max_tokens=1000):
    m = get_model(model_id)
    ok, hint = _status(m)
    if not ok:
        raise ProviderError(f"{m['label']} is not available: {hint}.")
    p = m["provider"]
    if p == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(model=model_id, max_tokens=max_tokens, system=system,
                                   messages=[{"role": "user", "content": user}])
        return r.content[0].text
    if p == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model_id, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content
    if p == "groq":
        from groq import Groq
        r = Groq().chat.completions.create(
            model=model_id, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content
    if p == "ollama":
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=300, json={
            "model": model_id, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise ProviderError("Unknown provider.")

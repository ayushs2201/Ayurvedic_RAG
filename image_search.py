"""Herb photo search (Wikimedia Commons) + optional photo identification."""
import base64, re
import requests

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "AyurvedaRAG-WILProject/1.0 (RMIT Master of Data Science; contact: ayushsrivastava363642@gmail.com)"}

_PATTERNS = [
    re.compile(r"\b(?:images?|photos?|pictures?|pics?)\s+(?:of|for)\s+(.+?)\s*[?.!]*$", re.I),
    re.compile(r"\bshow\s+(?:me\s+)?(?:a\s+|the\s+)?(.+?)\s+(?:images?|photos?|pictures?|pics?)\s*[?.!]*$", re.I),
    re.compile(r"\bwhat\s+does\s+(.+?)\s+look\s+like", re.I),
]


def extract_image_request(question):
    """Returns the herb name if the user is asking to SEE something, else None."""
    for p in _PATTERNS:
        m = p.search(question.strip())
        if m:
            return re.sub(r"^(?:the|a|an)\s+", "", m.group(1).strip(), flags=re.I)[:60]
    return None


def fetch_reference_photos(herb_name, max_results=4):
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"{herb_name} plant", "gsrnamespace": 6, "gsrlimit": max_results,
              "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 480}
    r = requests.get(WIKIMEDIA_API, params=params, timeout=15, headers=HEADERS)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if url:
            out.append({"title": page.get("title", "").replace("File:", ""), "url": url,
                        "full_url": info.get("url"), "description_url": info.get("descriptionurl")})
    return out


def identify_from_user_photo(image_bytes, media_type="image/jpeg", model="claude-sonnet-5"):
    """ASSISTIVE ONLY - never a confirmed identification."""
    import anthropic
    client = anthropic.Anthropic()
    data = base64.standard_b64encode(image_bytes).decode("utf-8")
    resp = client.messages.create(model=model, max_tokens=500, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
        {"type": "text", "text": (
            "Describe the distinguishing visual features of the plant/substance in this image (leaf shape, "
            "color, texture, growth pattern, etc). Suggest possible identifications only if reasonably "
            "confident, and explicitly state that this is not a confirmed identification - a qualified "
            "botanist or Ayurvedic practitioner should verify before any medicinal use or consumption.")}]}])
    return resp.content[0].text

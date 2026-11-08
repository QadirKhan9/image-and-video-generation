import io
import os
import requests
import urllib.parse
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

def query_pollinations(prompt: str, model_name: str = "flux") -> Image.Image:
    """Generate an image using Pollinations.ai API."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required")
    encoded = urllib.parse.quote(prompt.strip())
    url = f"https://gen.pollinations.ai/image/{encoded}"
    params = {
        "width": 1024,
        "height": 1024,
        "nologo": "true",
        "model": model_name,
    }
    if POLLINATIONS_API_KEY:
        params["key"] = POLLINATIONS_API_KEY.strip()
    resp = requests.get(url, params=params, timeout=45)
    if resp.status_code == 200:
        return Image.open(io.BytesIO(resp.content))
    elif resp.status_code == 402:
        try:
            err = resp.json().get("error", "Rate limit exceeded.")
        except Exception:
            err = "Rate limit exceeded (too many concurrent requests)."
        raise RuntimeError(f"Pollinations.ai error: {err}")
    else:
        raise RuntimeError(f"Pollinations.ai returned {resp.status_code}: {resp.text}")

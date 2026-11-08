import os
import io
import sys
import socket
import urllib.parse
import requests
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# DNS PATCH (kept from original for reliability)
# ---------------------------------------------------------------------------
try:
    import dns.resolver as _dns_resolver
    import dns.message as _dns_msg
    import dns.query as _dns_query
    import dns.rdatatype as _dns_rdt

    _original_getaddrinfo = socket.getaddrinfo
    _FALLBACK_DNS = "8.8.8.8"
    _dns_cache = {}

    def _resolve_via_google(host: str) -> str | None:
        if host in _dns_cache:
            return _dns_cache[host]
        try:
            q = _dns_msg.make_query(host, _dns_rdt.A)
            resp = _dns_query.udp(q, _FALLBACK_DNS, timeout=5)
            if resp.answer:
                ip = str(resp.answer[0][0])
                _dns_cache[host] = ip
                return ip
        except Exception:
            pass
        return None

    def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return _original_getaddrinfo(host, port, family, type, proto, flags)
        except socket.gaierror:
            ip = _resolve_via_google(host) if isinstance(host, str) else None
            if ip:
                return _original_getaddrinfo(ip, port, family, type, proto, flags)
            raise

    socket.getaddrinfo = _patched_getaddrinfo
    print("[DNS] Fallback DNS patch active (Google 8.8.8.8)")
except ImportError:
    print("[DNS] dnspython not installed; no DNS fallback patch.")

# ---------------------------------------------------------------------------
# Environment and configuration
# ---------------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
HF_SPACE_MODELS = {
    "FLUX.1 Schnell — Free Hugging Face Space": "black-forest-labs/FLUX.1-schnell",
}

MODELS = {
    "Pollinations.ai — Flux (Free & No Key)": "flux",
    "Pollinations.ai — ZImage": "zimage",
    **HF_SPACE_MODELS,
}

# ---------------------------------------------------------------------------
# Provider functions
# ---------------------------------------------------------------------------
def query_huggingface(prompt: str, model_name: str) -> Image.Image:
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt before generating.")
    if not HF_TOKEN:
        raise gr.Error("HF_TOKEN is missing. Add it to your .env file.")
    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    try:
        image = client.text_to_image(prompt=prompt.strip(), model=model_name)
        return image
    except Exception as e:
        err = str(e)
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 410 or "no longer supported by provider" in err.lower():
            raise gr.Error(
                "This model is no longer supported by Hugging Face Inference. "
                "Select another model, such as Pollinations.ai Flux."
            ) from e
        if status_code == 402 or "payment required" in err.lower():
            raise gr.Error(
                "Hugging Face Inference requires prepaid credits for this request. "
                "Add credits to your Hugging Face account, or select Pollinations.ai Flux or ZImage."
            ) from e
        if "401" in err or "unauthorized" in err.lower():
            raise gr.Error("Invalid HF_TOKEN.")
        if "429" in err or "rate limit" in err.lower():
            raise gr.Error("Rate limit reached. Try again later.")
        raise gr.Error(f"Generation failed: {err}")


def query_huggingface_space(prompt: str, space_id: str) -> Image.Image:
    """Generate an image through a public Hugging Face Space."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt before generating.")
    from gradio_client import Client

    try:
        client = Client(space_id, verbose=False)
        result, _ = client.predict(
            prompt=prompt.strip(),
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer",
        )
        if isinstance(result, dict):
            image_path = result.get("path")
        elif isinstance(result, (str, os.PathLike)):
            image_path = result
        else:
            image_path = None
        if not image_path:
            raise gr.Error("The Hugging Face Space did not return an image.")
        with Image.open(image_path) as image:
            return image.copy()
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(
            "The free Hugging Face Space is unavailable or busy. Please try again later."
        ) from e


def query_pollinations(prompt: str, model_name: str = "flux") -> Image.Image:
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")
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
    try:
        resp = requests.get(url, params=params, timeout=80)
    except requests.exceptions.Timeout:
        raise gr.Error("Pollinations.ai is taking too long to respond (server overload). Please try again.")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Network error when contacting Pollinations.ai: {e}")

    if resp.status_code == 200:
        return Image.open(io.BytesIO(resp.content))
    elif resp.status_code == 402:
        try:
            err = resp.json().get("error", "Rate limit exceeded.")
        except Exception:
            err = "Rate limit exceeded (too many concurrent requests)."
        raise gr.Error(f"Pollinations.ai error: {err}")
    else:
        raise gr.Error(f"Pollinations.ai returned {resp.status_code}: {resp.text}")

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def generate_image(prompt: str, model_name: str):
    """Generate and return an image without saving it to disk."""
    if model_name not in MODELS:
        raise gr.Error("Please select a supported image model.")
    model_id = MODELS[model_name]
    if model_name.startswith("Pollinations.ai"):
        image = query_pollinations(prompt, model_id)
    elif model_name in HF_SPACE_MODELS:
        image = query_huggingface_space(prompt, model_id)
    else:
        image = query_huggingface(prompt, model_id)
    return image

# ---------------------------------------------------------------------------
# Gradio UI (standalone – can be imported by app.py)
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Private AI Image Generator")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Prompt", placeholder="A futuristic city at night, neon lights...", lines=4)
            model_dropdown = gr.Dropdown(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label="Select AI Model")
            generate_btn = gr.Button("Generate Image", variant="primary")
        with gr.Column(scale=2):
            image_output = gr.Image(label="Generated Result", type="pil")
    generate_btn.click(fn=generate_image, inputs=[prompt_input, model_dropdown], outputs=image_output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), show_error=True)

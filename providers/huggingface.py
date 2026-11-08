import os
import io
from PIL import Image
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

def query_huggingface(prompt: str, model_id: str) -> Image.Image:
    """Generate an image using HuggingFace InferenceClient."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt before generating.")
    if not HF_TOKEN:
        raise gr.Error("HF_TOKEN is missing. Add it to your .env file.")
    try:
        image = client.text_to_image(prompt=prompt.strip(), model=model_id)
        return image
    except Exception as e:
        err = str(e)
        if "401" in err or "unauthorized" in err.lower():
            raise gr.Error("Invalid HF_TOKEN.")
        if "429" in err or "rate limit" in err.lower():
            raise gr.Error("Rate limit reached. Try again later.")
        raise gr.Error(f"Generation failed: {err}")

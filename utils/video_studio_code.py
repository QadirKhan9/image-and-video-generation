import os
import gradio as gr
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load environment variables from .env (local dev only – HF Spaces use Secrets)
load_dotenv()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
VIDEO_MODELS = {
    "LTX-2 T2V - High Quality (Text-to-Video)": {
        "space": "Lightricks/LTX-2-3",
        "task":  "t2v",
        "mode":  "sync",
    },
    "AnimateDiff Lightning T2V - Fast (Text-to-Video)": {
        "space": "ByteDance/AnimateDiff-Lightning",
        "task":  "t2v",
        "mode":  "sync",
    },
    "LTX-2 I2V - High Quality (Image-to-Video)": {
        "space": "Lightricks/LTX-2-3",
        "task":  "i2v",
        "mode":  "sync",
    },
    "Stable Video Diffusion I2V (Image-to-Video)": {
        "space": "multimodalart/stable-video-diffusion",
        "task":  "i2v",
        "mode":  "sync",
    },
}

# ---------------------------------------------------------------------------
# Generation Functions
# ---------------------------------------------------------------------------

def _generate_animatediff(client, prompt):
    """AnimateDiff Lightning - synchronous, fast."""
    result = client.predict(
        prompt=prompt.strip(),
        base="ToonYou",
        motion="",
        step=4,
        api_name="/generate_image",
    )
    video_path = result['video']
    if isinstance(video_path, dict):
        video_path = video_path.get('path', video_path)
    return video_path


def _generate_svd(client, input_image):
    """Stable Video Diffusion - synchronous."""
    result = client.predict(
        image=handle_file(input_image),
        seed=0,
        randomize_seed=True,
        motion_bucket_id=127,
        fps_id=6,
        api_name="/video",
    )
    video_path = result[0]['video']
    if isinstance(video_path, dict):
        video_path = video_path.get('path', video_path)
    return video_path


def _generate_ltx2(client, input_image, prompt):
    """LTX-2 Text-to-Video / Image-to-Video - synchronous, high quality."""
    result = client.predict(
        input_image=handle_file(input_image) if input_image else None,
        prompt=prompt.strip() if prompt else "Make this image come alive with cinematic motion",
        duration=3.0,
        enhance_prompt=True,
        seed=10,
        randomize_seed=True,
        height=1024,
        width=1536,
        api_name="/generate_video",
    )
    video_path = result[0]
    if isinstance(video_path, dict):
        video_path = video_path.get('path', video_path)
    return video_path

# ---------------------------------------------------------------------------
# Main Dispatcher
# ---------------------------------------------------------------------------

def generate_video(prompt, input_image, model_name):
    """Dispatch the selected generator and return its temporary result path."""
    # Use HF_VIDEO_TOKEN for video generation (separate account = separate ZeroGPU quota)
    # Falls back to HF_TOKEN if HF_VIDEO_TOKEN is not set
    HF_TOKEN = os.getenv("HF_VIDEO_TOKEN") or os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise gr.Error("Missing HF_VIDEO_TOKEN (or HF_TOKEN). Add it as an environment variable or Space Secret.")

    cfg  = VIDEO_MODELS[model_name]
    task = cfg["task"]

    if task == "i2v" and not input_image:
        raise gr.Error("Please upload an initial image for Image-to-Video models.")
    if task == "t2v" and not (prompt and prompt.strip()):
        raise gr.Error("Please enter a prompt for Text-to-Video models.")

    try:
        try:
            client = Client(cfg["space"], hf_token=HF_TOKEN, httpx_kwargs={"timeout": 600})
        except TypeError:
            # gradio_client version >= 1.0.0 renamed 'hf_token' to 'token'
            client = Client(cfg["space"], token=HF_TOKEN, httpx_kwargs={"timeout": 600})  # type: ignore

        if "AnimateDiff" in model_name:
            raw_path = _generate_animatediff(client, prompt)
        elif "LTX-2" in model_name:
            raw_path = _generate_ltx2(client, input_image, prompt)
        elif "Stable Video" in model_name:
            raw_path = _generate_svd(client, input_image)
        else:
            raise gr.Error("Unsupported video model.")

    except gr.Error:
        raise
    except Exception as e:
        err = str(e)
        if "401" in err or "unauthorized" in err.lower():
            raise gr.Error("Invalid HF_TOKEN. Please check your token.")
        if "quota" in err.lower() or "exceeded" in err.lower():
            raise gr.Error("HuggingFace ZeroGPU quota exceeded. Try again later.")
        if "no gpu" in err.lower():
            raise gr.Error("HuggingFace Spaces: No free GPUs available right now. Please try again in a few minutes.")
        raise gr.Error(f"Video generation failed: {err}")

    return raw_path

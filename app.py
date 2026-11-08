import os
import sys
import asyncio
from typing import Callable

# Fix: Windows ProactorEventLoop throws WinError 10054 when a remote host
# closes the connection. Switching to SelectorEventLoop avoids this noise.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr
from gradio.themes import Soft
from dotenv import load_dotenv

# Ensure project root is on PYTHONPATH for relative imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import core generation functions
# pyrefly: ignore [missing-import]
from utils.ai_studio_code import generate_image, MODELS
from utils.video_studio_code import generate_video, VIDEO_MODELS
load_dotenv()

# ---------------------------------------------------------------------------
# UI Layout – Premium Tabbed Interface
# ---------------------------------------------------------------------------
with gr.Blocks() as app:
    gr.Markdown("# 🎨 Private AI Studio – Image & Video Generation")
    with gr.Tabs():
        # ---------------- Image Generation Tab ----------------
        with gr.TabItem("🖼️ Image Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="A futuristic city at night, neon lights, cinematic...",
                        lines=4,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Pollinations.ai — Flux (Free & No Key)",
                        label="Select AI Model",
                    )
                    generate_btn = gr.Button("Generate Image", variant="primary")
                with gr.Column(scale=2):
                    image_output = gr.Image(label="Generated Result", type="pil")
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt_input, model_dropdown],
                outputs=image_output,
                api_name="generate_image"
            )

        # ---------------- Video Generation Tab ----------------
        with gr.TabItem("🎥 Video Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_prompt = gr.Textbox(
                        label="Prompt (required for Text‑to‑Video)",
                        placeholder="A cinematic drone shot over a futuristic city at night...",
                        lines=3,
                    )
                    video_image_input = gr.Image(
                        label="Initial Image (required for Image‑to‑Video)",
                        type="filepath",
                    )
                    video_model_dropdown = gr.Dropdown(
                        choices=list(VIDEO_MODELS.keys()),
                        value=list(VIDEO_MODELS.keys())[0],
                        label="Select Video Model",
                    )
                    video_generate_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Generated Video Result")

            video_generate_btn.click(
                fn=generate_video,
                inputs=[video_prompt, video_image_input, video_model_dropdown],
                outputs=video_output,
                api_name="generate_video"
            )


if __name__ == "__main__":
    auth_username = os.getenv("GRADIO_USERNAME")
    auth_password = os.getenv("GRADIO_PASSWORD")
    is_huggingface_space = bool(os.getenv("SPACE_ID"))

    if bool(auth_username) != bool(auth_password):
        raise RuntimeError("Set both GRADIO_USERNAME and GRADIO_PASSWORD.")
    if is_huggingface_space and not auth_username:
        raise RuntimeError(
            "Set GRADIO_USERNAME and GRADIO_PASSWORD in the Hugging Face Space secrets."
        )

    auth: Callable[[str, str], bool] | tuple[str, str] | list[tuple[str, str]] | None = (auth_username, auth_password) if auth_username else None # pyright: ignore[reportAssignmentType]
    app.launch(theme=Soft(), show_error=True, auth=auth)

import os
import sys
import asyncio

# Fix: Windows ProactorEventLoop throws WinError 10054 when a remote host
# closes the connection. Switching to SelectorEventLoop avoids this noise.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr
from gradio.themes import Soft

# Ensure project root is on PYTHONPATH for relative imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import core generation functions
# pyrefly: ignore [missing-import]
from utils.ai_studio_code import generate_image, MODELS
from utils.video_studio_code import generate_video, VIDEO_MODELS

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

            def _generate_image_with_controls(prompt, model):
                yield (
                    gr.skip(),
                    gr.update(value="Generating Image...", interactive=False),
                    gr.update(interactive=False),
                )
                try:
                    image = generate_image(prompt, model)
                except Exception:
                    yield (
                        gr.skip(),
                        gr.update(value="Generate Image", interactive=True),
                        gr.update(interactive=True),
                    )
                    raise
                yield (
                    image,
                    gr.update(value="Generate Image", interactive=True),
                    gr.update(interactive=True),
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

            def _generate_video_with_controls(prompt, image, model):
                yield (
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.update(value="Generating Video...", interactive=False),
                )
                try:
                    video = generate_video(prompt, image, model)
                except Exception:
                    yield (
                        gr.skip(),
                        gr.update(interactive=True),
                        gr.update(value="Generate Video", interactive=True),
                    )
                    raise
                yield (
                    video,
                    gr.update(interactive=True),
                    gr.update(value="Generate Video", interactive=True),
                )

            video_generate_btn.click(
                fn=_generate_video_with_controls,
                inputs=[video_prompt, video_image_input, video_model_dropdown],
                outputs=[video_output, generate_btn, video_generate_btn],
                api_name="generate_video",
                trigger_mode="once",
                concurrency_limit=1,
                concurrency_id="generation",
            )
            generate_btn.click(
                fn=_generate_image_with_controls,
                inputs=[prompt_input, model_dropdown],
                outputs=[image_output, generate_btn, video_generate_btn],
                api_name="generate_image",
                trigger_mode="once",
                concurrency_limit=1,
                concurrency_id="generation",
            )


if __name__ == "__main__":
    app.launch(
        theme=Soft(),
        show_error=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )

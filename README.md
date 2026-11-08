---
title: Private AI Studio
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.11"
app_file: app.py
---

# Private AI Studio

Image and video generation through Pollinations and Hugging Face Spaces.

This Space is public, but the app requires login. Add these values in
**Settings → Secrets** before the Space starts:

- `GRADIO_USERNAME` and `GRADIO_PASSWORD` to access the app.
- `HF_VIDEO_TOKEN` or `HF_TOKEN` for video generation.
- `POLLINATIONS_API_KEY` is optional.

Never commit API keys or passwords to this repository.

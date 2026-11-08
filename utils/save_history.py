import os
import json
from datetime import datetime

# Paths – adjust if project layout changes
def get_paths(base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))):
    images_dir = os.path.join(base_dir, "images")
    history_dir = os.path.join(base_dir, "history")
    metadata_path = os.path.join(history_dir, "image_metadata.json")
    return images_dir, history_dir, metadata_path


def build_image_metadata():
    """Scan the ``images`` folder and generate/update a JSON metadata file.

    The metadata file ``history/image_metadata.json`` contains a list of dicts:

    ```json
    [
        {
            "filename": "20240615_101530_my_prompt.png",
            "timestamp": "2024-06-15T10:15:30",
            "size_bytes": 123456,
            "prompt": "my prompt",
            "model": "Pollinations.ai — Flux (Free & No Key)"
        },
        ...
    ]
    ```

    Existing entries (already written by ai_studio_code.py with full prompt + model)
    are preserved. New files not yet in the metadata are appended with best-effort
    prompt extraction and model set to "unknown".
    """
    images_dir, history_dir, metadata_path = get_paths()
    os.makedirs(history_dir, exist_ok=True)

    # Load existing metadata so we don't overwrite prompt/model info
    existing = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    existing[item["filename"]] = item
        except Exception as e:
            print(f"[WARN] Could not read existing metadata: {e}")

    metadata = []
    if os.path.isdir(images_dir):
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # Reuse existing entry if available (has real prompt + model)
            if fname in existing:
                metadata.append(existing[fname])
                continue

            file_path = os.path.join(images_dir, fname)
            try:
                stat = os.stat(file_path)
                parts = fname.split("_")  # ['20260615', '041744', 'lion', 'running.png']
                # Timestamp occupies first 2 parts (YYYYMMDD + HHMMSS)
                ts_str = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").isoformat()
                except Exception:
                    ts = datetime.fromtimestamp(stat.st_mtime).isoformat()
                # Prompt is everything after the first 2 timestamp tokens, minus the extension
                prompt_raw = "_".join(parts[2:]).rsplit(".", 1)[0] if len(parts) > 2 else ""
                prompt = prompt_raw.replace("_", " ")
                metadata.append({
                    "filename": fname,
                    "timestamp": ts,
                    "size_bytes": stat.st_size,
                    "prompt": prompt,
                    "model": "unknown",
                })
            except Exception as e:
                print(f"[WARN] Could not process {fname}: {e}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Image metadata written to {metadata_path} ({len(metadata)} entries)")


if __name__ == "__main__":
    build_image_metadata()

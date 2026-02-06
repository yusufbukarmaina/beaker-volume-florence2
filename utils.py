# modules/utils.py
"""
Shared utilities for Florence-2 modules (updated for Beaker Volume VQA)
"""
import os
import re
import time
import io
import json
import logging
from pathlib import Path
from threading import Thread
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login

# ---------------- Logging ----------------
def setup_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"florence2_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("florence2"), log_file


logger, _ = setup_logging()

# ---------------- Visualization ----------------
COLORMAP = [
    "red", "blue", "green", "orange", "purple",
    "brown", "pink", "gray", "olive", "cyan"
]

# ---------------- Tasks ----------------
# Keep demo tasks + add beaker volume
TASK_PROMPTS = {
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>",

    # NEW: numeric VQA task for beaker volume
    "Beaker Volume (mL)": "What is the liquid volume in mL? Answer with a number only."
}

# Regex for float extraction (robust)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def extract_number(text: str) -> Optional[float]:
    """Extract first float-like number from a string; return None if not found."""
    if not text:
        return None
    s = text.replace(",", " ")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None

def overlay_text_on_image(img: Image.Image, lines, box_alpha: float = 0.75) -> Image.Image:
    """Overlay text box (bottom-left) on an image."""
    if not isinstance(lines, (list, tuple)):
        lines = [str(lines)]
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try a default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text = "\n".join([str(x) for x in lines])
    padding = 10

    # measure text
    bbox = draw.multiline_textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = padding, img.height - th - 3 * padding

    # background box
    box = Image.new("RGBA", (tw + 2 * padding, th + 2 * padding), (255, 255, 255, int(255 * box_alpha)))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(box, (x - padding, y - padding), box)
    img = img_rgba.convert("RGB")

    draw = ImageDraw.Draw(img)
    draw.multiline_text((x, y), text, fill=(0, 0, 0), font=font)
    return img

def visualize_object_detection(image, prediction, image_format="pil"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for i, (bbox, label) in enumerate(zip(prediction["bboxes"], prediction["labels"])):
        color = COLORMAP[i % len(COLORMAP)]
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        plt.text(x1, y1 - 5, label, color="white", fontsize=12,
                 bbox=dict(facecolor=color, alpha=0.7))

    ax.axis("off")
    plt.title(f"Detected {len(prediction['labels'])} objects", fontsize=16)

    if image_format == "pil":
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    else:
        plt.tight_layout()
        return fig

def visualize_caption(image, caption, image_format="pil"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")

    caption_text = caption if isinstance(caption, str) else str(caption)
    ax.text(0.5, -0.1, caption_text, wrap=True, fontsize=12,
            ha="center", va="top", transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    if image_format == "pil":
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    else:
        return fig

# ---------------- Model Manager ----------------
class ModelManager:
    """Manager class for loading and caching models/processors."""
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.logger, _ = setup_logging()

    def get_model(self, model_id: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = f"{model_id}_{device}"
        if cache_key in self.models:
            return self.models[cache_key], self.processors[cache_key]

        self.logger.info(f"Loading model: {model_id} on {device}")
        dtype = torch.float16 if (device == "cuda") else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype
        ).eval().to(device)

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self.models[cache_key] = model
        self.processors[cache_key] = processor
        self.logger.info(f"Loaded model: {model_id}")
        return model, processor

    def authenticate_hub(self, token: str) -> bool:
        try:
            login(token=token)
            self.logger.info("Authenticated with Hugging Face Hub")
            return True
        except Exception as e:
            self.logger.error(f"Failed to authenticate: {str(e)}")
            return False

model_manager = ModelManager()

def stream_logs(file_path: str, output_fn):
    def _stream():
        while not os.path.exists(file_path):
            time.sleep(0.5)
        with open(file_path, "r") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    output_fn(line)
                else:
                    time.sleep(0.1)

    t = Thread(target=_stream, daemon=True)
    t.start()
    return t

class ProgressMonitor:
    """Helper class to monitor and update progress in UI"""
    def __init__(self, progress=None, status=None):
        self.progress = progress
        self.status = status

    def update(self, current, total, status_text):
        if self.progress is not None:
            try:
                self.progress(value=current / total, desc=status_text)
            except Exception:
                pass
        if self.status is not None:
            try:
                self.status(status_text)
            except Exception:
                pass
        return current / total, status_text

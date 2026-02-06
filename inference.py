# modules/inference.py
"""
Florence-2 inference module (updated for Beaker Volume VQA)
"""
import time
from typing import Dict, Any, Tuple, Union, Optional

import torch
from PIL import Image

from .utils import (
    model_manager,
    visualize_object_detection,
    visualize_caption,
    TASK_PROMPTS,
    setup_logging,
    extract_number,
    overlay_text_on_image
)

logger, _ = setup_logging()

def _prepare_inputs(processor, image: Image.Image, text: str, device, model_dtype):
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=text, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)  # keep int
    pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, pixel_values, attention_mask

def run_inference(
    image: Image.Image,
    task_name: str,
    model_id: str = "microsoft/Florence-2-base-ft",
    device: str = None
) -> Tuple[Dict[str, Any], float]:
    """
    Run inference for Florence built-in tasks (caption etc.)
    """
    if task_name not in TASK_PROMPTS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_PROMPTS.keys())}")

    task_prompt = TASK_PROMPTS[task_name]
    model, processor = model_manager.get_model(model_id, device)
    device_t = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    input_ids, pixel_values, attention_mask = _prepare_inputs(processor, image, task_prompt, device_t, model_dtype)

    start = time.time()
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=3,
            max_new_tokens=256,
        )
    infer_time = time.time() - start

    # Florence demo decoding keeps tokens
    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    cleaned = gen_text.replace("</s>", "").replace("<s>", "").replace("<pad>", "")

    parsed = processor.post_process_generation(
        cleaned,
        task=task_prompt,
        image_size=(image.width, image.height),
    )

    return parsed, infer_time

def run_volume_inference(
    image: Image.Image,
    model_id: str,
    prompt: Optional[str] = None,
    device: str = None,
    num_beams: int = 3,
    max_new_tokens: int = 16,
) -> Tuple[Dict[str, Any], float]:
    """
    Run numeric VQA: returns raw_text + parsed pred_volume_ml
    """
    if prompt is None:
        prompt = TASK_PROMPTS["Beaker Volume (mL)"]

    model, processor = model_manager.get_model(model_id, device)
    device_t = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    input_ids, pixel_values, attention_mask = _prepare_inputs(processor, image, prompt, device_t, model_dtype)

    start = time.time()
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    infer_time = time.time() - start

    raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    pred = extract_number(raw)

    return {"raw_text": raw, "pred_volume_ml": pred}, infer_time

def process_result(image: Image.Image, result: Dict[str, Any], task_name: str) -> Union[Image.Image, str]:
    """
    Display helper for Gradio.
    For volume: overlay prediction text on image.
    """
    if task_name == "Beaker Volume (mL)":
        pred = result.get("pred_volume_ml", None)
        raw = result.get("raw_text", "")
        lines = [
            f"Predicted: {pred} mL" if pred is not None else "Predicted: (parse failed)",
            f"Raw: {raw}"
        ]
        return overlay_text_on_image(image, lines)

    task_prompt = TASK_PROMPTS[task_name]

    # (OD not used in your UI now, but kept for compatibility)
    if task_name == "Object Detection":
        det = result.get(task_prompt, {})
        return visualize_object_detection(image, det)

    caption = result.get(task_prompt, "")
    return visualize_caption(image, caption)

# modules/evaluation.py
"""
Florence-2 evaluation module (updated for Beaker Volume VQA metrics)
"""
from typing import Dict, Any, Optional
import numpy as np
import torch
from datasets import load_dataset

from .utils import model_manager, setup_logging, extract_number, TASK_PROMPTS
from .inference import run_volume_inference

logger, _ = setup_logging()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_dataset(
    dataset_name: str,
    split: str,
    model_id: str,
    prompt_key: str = "prompt",
    gt_key: str = "label",        # can be "label" or "volume_ml"
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Dataset-level evaluation for beaker volume.
    Returns metrics + per-row prediction list (for saving to CSV later if you want).
    """
    ds = load_dataset(dataset_name)
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    data = ds[split]
    n = len(data) if max_samples is None else min(len(data), max_samples)

    rows = []
    y_true, y_pred = [], []
    skipped = 0

    for i in range(n):
        ex = data[i]
        image = ex["image"]
        prompt = ex.get(prompt_key, TASK_PROMPTS["Beaker Volume (mL)"])

        gt_val = ex.get(gt_key, None)
        if gt_val is None:
            gt_num = None
        elif isinstance(gt_val, (int, float)):
            gt_num = float(gt_val)
        else:
            gt_num = extract_number(str(gt_val))

        out, _ = run_volume_inference(image=image, model_id=model_id, prompt=prompt)
        pred = out.get("pred_volume_ml", None)

        if gt_num is None or pred is None:
            skipped += 1
        else:
            y_true.append(gt_num)
            y_pred.append(float(pred))

        rows.append({
            "idx": i,
            "image_name": ex.get("image_name"),
            "beaker_ml": ex.get("beaker_ml"),
            "view": ex.get("view"),
            "condition": ex.get("condition"),
            "gt_volume_ml": gt_num,
            "pred_volume_ml": pred,
            "raw_text": out.get("raw_text", "")
        })

    y_true = np.array(y_true, dtype=float) if len(y_true) else np.array([])
    y_pred = np.array(y_pred, dtype=float) if len(y_pred) else np.array([])

    metrics = compute_metrics(y_true, y_pred) if len(y_true) else {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    return {
        "dataset": dataset_name,
        "split": split,
        "model_id": model_id,
        "n_total": n,
        "n_scored": int(len(y_true)),
        "n_skipped": int(skipped),
        "metrics": metrics,
        "rows": rows
    }

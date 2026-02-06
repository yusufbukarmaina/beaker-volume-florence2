# modules/training.py
"""
Florence-2 training module (updated for Beaker Volume VQA)
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .utils import model_manager, setup_logging, ProgressMonitor

logger, log_file = setup_logging()

class MultiTaskDataset(Dataset):
    """
    For beaker dataset: expects columns: image, prompt, label
    label should be text like "32.5" (number only recommended)
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = ex["prompt"]
        label = ex["label"]
        image = ex["image"]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        return prompt, label, image

def collate_fn(batch, processor, max_length: int = 256):
    prompts, targets, images = zip(*batch)
    inputs = processor(
        text=list(prompts),
        images=list(images),
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return inputs, list(targets)

def load_sharded_dataset(dataset_name: str):
    all_splits = load_dataset(dataset_name)

    train_splits = []
    if "train" in all_splits:
        train_splits.append(all_splits["train"])
    for k in all_splits.keys():
        if k.startswith("train_shard_"):
            train_splits.append(all_splits[k])

    if not train_splits:
        raise ValueError("No training splits found (train or train_shard_*)")

    train_full = concatenate_datasets(train_splits)

    if "val" not in all_splits:
        raise ValueError("No validation split found: expected 'val'")

    val_full = all_splits["val"]
    logger.info(f"Train: {len(train_full)} | Val: {len(val_full)}")
    return {"train": train_full, "val": val_full}

class TrainingMonitor:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        self.best_val = float("inf")

    def update(self, train_loss, val_loss, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_history.append(lr)
        improved = val_loss < self.best_val
        if improved:
            self.best_val = val_loss
        return improved

    def plot(self, epoch: int):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train")
        plt.plot(self.val_losses, label="Val")
        plt.title("Loss vs Epoch")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.lr_history)
        plt.title("LR vs Epoch")
        plt.grid(True)

        out = self.save_dir / f"metrics_epoch_{epoch}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return str(out)

def _move_inputs_to_device(inputs, device, model_dtype):
    input_ids = inputs["input_ids"].to(device)  # keep int
    pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, pixel_values, attention_mask

def train_model(
    dataset_name: str,
    base_model_id: str,
    output_dir: str,
    config: Dict,
    progress_monitor: Optional[ProgressMonitor] = None,
) -> str:
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config.get("batch_size", 2))
    learning_rate = float(config.get("learning_rate", 2e-5))
    epochs = int(config.get("epochs", 5))
    save_frequency = int(config.get("save_frequency", 1))
    num_workers = int(config.get("num_workers", 2))
    freeze_vision = bool(config.get("freeze_vision", True))
    max_length = int(config.get("max_length", 256))
    grad_accum = int(config.get("grad_accum_steps", 1))

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if progress_monitor:
        progress_monitor.update(0, 1, "Loading dataset...")

    dataset = load_sharded_dataset(dataset_name)
    train_dataset = MultiTaskDataset(dataset["train"])
    val_dataset = MultiTaskDataset(dataset["val"])

    if progress_monitor:
        progress_monitor.update(0.1, 1, "Loading model...")

    model, processor = model_manager.get_model(base_model_id, device)
    model_dtype = next(model.parameters()).dtype

    # Optional freeze vision
    if freeze_vision and hasattr(model, "vision_tower"):
        logger.info("Freezing vision tower parameters")
        for p in model.vision_tower.parameters():
            p.requires_grad = False
    else:
        logger.info("Not freezing vision tower (or attribute not found)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, processor, max_length=max_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, processor, max_length=max_length),
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * max(1, len(train_loader) // grad_accum)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Stability: train in float32 if you want (recommended when NaNs happen)
    if config.get("force_fp32", True):
        logger.info("Forcing model to float32 for stability")
        model = model.to(torch.float32)
        model_dtype = next(model.parameters()).dtype

    monitor = TrainingMonitor(save_dir / "metrics")

    logger.info(f"Device={device} | dtype={model_dtype} | bs={batch_size} | accum={grad_accum} | epochs={epochs}")

    if progress_monitor:
        progress_monitor.update(0.2, 1, "Training started...")

    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0

        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids, pixel_values, attention_mask = _move_inputs_to_device(inputs, device, model_dtype)

            labels = processor.tokenizer(
                text=list(targets),
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            if torch.isnan(loss):
                logger.warning("NaN loss detected. Skipping batch.")
                continue

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += float(loss.item())
            seen += 1

            if progress_monitor:
                # progress across epochs
                base = 0.2 + (epoch / epochs) * 0.7
                frac = (step + 1) / max(1, len(train_loader))
                prog = base + frac * (0.7 / epochs)
                progress_monitor.update(prog, 1, f"Epoch {epoch+1}/{epochs} step {step+1}/{len(train_loader)} loss {loss.item():.4f}")

        avg_train = total_loss / max(1, seen)

        # Validation
        model.eval()
        val_loss = 0.0
        vseen = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                input_ids, pixel_values, attention_mask = _move_inputs_to_device(inputs, device, model_dtype)

                labels = processor.tokenizer(
                    text=list(targets),
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                out = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                if torch.isnan(out.loss):
                    continue
                val_loss += float(out.loss.item())
                vseen += 1

        avg_val = val_loss / max(1, vseen)
        logger.info(f"Epoch {epoch+1}: train={avg_train:.4f} | val={avg_val:.4f}")

        improved = monitor.update(avg_train, avg_val, optimizer.param_groups[0]["lr"])
        monitor.plot(epoch + 1)

        # Save best + checkpoint
        if improved:
            best_dir = save_dir / "best_model"
            best_dir.mkdir(exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

        if (epoch + 1) % save_frequency == 0:
            ckpt = save_dir / f"checkpoint_epoch_{epoch+1}"
            ckpt.mkdir(exist_ok=True)
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)

    final_dir = save_dir / "final_model"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    if progress_monitor:
        progress_monitor.update(1, 1, "Training completed!")

    return str(final_dir)

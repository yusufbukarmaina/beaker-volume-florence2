#!/usr/bin/env python3
"""
Florence-2 Gradio App (updated for Beaker Volume Estimation)
"""
import os
import time
import torch
import gradio as gr

from modules import inference, training, evaluation, utils

logger, log_file = utils.setup_logging()

MODEL_CHOICES = [
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft"
]

TASK_CHOICES = [
    "Beaker Volume (mL)",
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
]

def clear_gpu_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            if hasattr(utils.model_manager, "models"):
                utils.model_manager.models.clear()
            if hasattr(utils.model_manager, "processors"):
                utils.model_manager.processors.clear()
            return "GPU memory cleared successfully!"
        return "No GPU available."
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {str(e)}")
        return f"Error clearing memory: {str(e)}"

def inference_router(image, task, model_id):
    if image is None:
        return None, "Please upload an image."

    try:
        if task == "Beaker Volume (mL)":
            out, t = inference.run_volume_inference(image=image, model_id=model_id)
            vis = inference.process_result(image, out, task)
            pred = out.get("pred_volume_ml", None)
            raw = out.get("raw_text", "")
            msg = f"Predicted volume: {pred} mL | time: {t:.2f}s | raw: {raw}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return vis, msg

        # caption tasks
        results, t = inference.run_inference(image, task, model_id)
        vis = inference.process_result(image, results, task)
        msg = f"Done in {t:.2f}s"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return vis, msg

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, f"Error: {str(e)}"

def train_fn(dataset_name, base_model, epochs, batch_size, learning_rate, freeze_vision, progress=gr.Progress()):
    try:
        out_dir = f"outputs/finetuned_{int(time.time())}"
        os.makedirs(out_dir, exist_ok=True)

        monitor = utils.ProgressMonitor(progress=progress, status=None)

        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "save_frequency": 1,
            "num_workers": 2,
            "freeze_vision": bool(freeze_vision),

            # recommended
            "max_length": 256,
            "grad_accum_steps": 1,
            "force_fp32": True,
        }

        model_path = training.train_model(
            dataset_name=dataset_name,
            base_model_id=base_model,
            output_dir=out_dir,
            config=config,
            progress_monitor=monitor,
        )

        return model_path, f"Training completed! Model saved to {model_path}"
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return None, f"Error: {str(e)}"

def eval_dataset_fn(dataset_name, split, model_id, gt_key, max_samples):
    if not dataset_name:
        return "Please provide dataset name."
    if not model_id:
        return "Please provide model path or Hub ID."

    try:
        max_samples = int(max_samples) if max_samples else None
        report = evaluation.evaluate_dataset(
            dataset_name=dataset_name,
            split=split,
            model_id=model_id,
            gt_key=gt_key,
            max_samples=max_samples
        )
        m = report["metrics"]
        return (
            f"Dataset: {report['dataset']} ({report['split']})\n"
            f"Model: {report['model_id']}\n"
            f"Total: {report['n_total']} | Scored: {report['n_scored']} | Skipped: {report['n_skipped']}\n\n"
            f"MAE:  {m['MAE']:.4f}\n"
            f"RMSE: {m['RMSE']:.4f}\n"
            f"R²:   {m['R2']:.4f}\n"
        )
    except Exception as e:
        logger.error(f"Eval error: {str(e)}")
        return f"Error: {str(e)}"

def upload_to_hub_fn(model_path, repo_id, token):
    try:
        ok = utils.model_manager.authenticate_hub(token)
        if not ok:
            return "Authentication failed. Check your token."

        from huggingface_hub import create_repo, upload_folder
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)

        upload_folder(folder_path=model_path, repo_id=repo_id, repo_type="model", token=token)
        return f"Uploaded successfully to {repo_id}"
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return f"Error: {str(e)}"

def create_app():
    with gr.Blocks(title="Beaker Volume Florence-2") as app:
        gr.Markdown("# Florence-2 – Beaker Volume Estimation (mL)")
        gr.Markdown("Train, infer and evaluate Florence-2 for liquid volume estimation in beakers.")

        with gr.Tabs():
            # ---------------- Inference ----------------
            with gr.Tab("Inference"):
                gr.Markdown("## Single Image Inference")
                with gr.Row():
                    with gr.Column():
                        inf_image = gr.Image(type="pil", label="Upload Image")
                        inf_task = gr.Dropdown(choices=TASK_CHOICES, value=TASK_CHOICES[0], label="Task")
                        inf_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
                        inf_clear = gr.Button("Clear GPU Memory")
                        inf_btn = gr.Button("Run Inference")
                    with gr.Column():
                        inf_output = gr.Image(label="Output")
                        inf_msg = gr.Textbox(label="Status", lines=4)

                inf_clear.click(fn=clear_gpu_memory, inputs=[], outputs=[inf_msg])
                inf_btn.click(fn=inference_router, inputs=[inf_image, inf_task, inf_model], outputs=[inf_output, inf_msg])

            # ---------------- Training ----------------
            with gr.Tab("Training"):
                gr.Markdown("## Fine-tune on Your Beaker Dataset (HF)")
                with gr.Row():
                    with gr.Column():
                        train_dataset = gr.Textbox(label="Dataset Name (HF)", placeholder="username/dataset-name")
                        train_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Base Model")
                        train_epochs = gr.Slider(1, 25, value=5, step=1, label="Epochs")
                        train_bs = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                        train_lr = gr.Slider(1e-6, 1e-4, value=2e-6, step=1e-7, label="Learning Rate")
                        train_freeze = gr.Checkbox(value=True, label="Freeze Vision Tower")
                        train_btn = gr.Button("Start Training")
                    with gr.Column():
                        train_out = gr.Textbox(label="Trained Model Path")
                        train_msg = gr.Textbox(label="Status", lines=4)

                train_btn.click(
                    fn=train_fn,
                    inputs=[train_dataset, train_model, train_epochs, train_bs, train_lr, train_freeze],
                    outputs=[train_out, train_msg],
                )

                with gr.Accordion("Upload to Hugging Face Hub", open=False):
                    hub_model_path = gr.Textbox(label="Model Path (local)")
                    hub_repo_id = gr.Textbox(label="Repo ID", placeholder="username/model-name")
                    hub_token = gr.Textbox(label="HF Token", type="password")
                    hub_btn = gr.Button("Upload")
                    hub_status = gr.Textbox(label="Upload Status", lines=3)

                    hub_btn.click(fn=upload_to_hub_fn, inputs=[hub_model_path, hub_repo_id, hub_token], outputs=[hub_status])

                train_out.change(fn=lambda x: x, inputs=train_out, outputs=hub_model_path)

            # ---------------- Evaluation ----------------
            with gr.Tab("Evaluation"):
                gr.Markdown("## Evaluate Model")

                with gr.Accordion("Single Image Test", open=True):
                    with gr.Row():
                        with gr.Column():
                            eval_image = gr.Image(type="pil", label="Upload Image")
                            eval_task = gr.Dropdown(choices=TASK_CHOICES, value="Beaker Volume (mL)", label="Task")
                            eval_model = gr.Textbox(label="Model Path or Hub ID", placeholder="path/to/model or username/model")
                            eval_clear = gr.Button("Clear GPU Memory")
                            eval_btn = gr.Button("Run")
                        with gr.Column():
                            eval_out = gr.Image(label="Output")
                            eval_msg = gr.Textbox(label="Status", lines=4)

                    eval_clear.click(fn=clear_gpu_memory, inputs=[], outputs=[eval_msg])

                    def eval_single_router(image, task, model_id):
                        return inference_router(image, task, model_id)

                    eval_btn.click(fn=eval_single_router, inputs=[eval_image, eval_task, eval_model], outputs=[eval_out, eval_msg])

                with gr.Accordion("Dataset Evaluation (MAE/RMSE/R²)", open=False):
                    eval_dataset = gr.Textbox(label="Dataset Name (HF)", placeholder="username/dataset-name")
                    eval_split = gr.Dropdown(choices=["val", "test", "train"], value="val", label="Split")
                    eval_model2 = gr.Textbox(label="Model Path or Hub ID", placeholder="path/to/model or username/model")
                    eval_gt_key = gr.Dropdown(choices=["label", "volume_ml"], value="label", label="Ground Truth Column")
                    eval_max = gr.Textbox(label="Max Samples (optional)", placeholder="e.g. 200 (leave empty for all)")
                    eval_run = gr.Button("Run Dataset Evaluation")
                    eval_report = gr.Textbox(label="Report", lines=10)

                    eval_run.click(
                        fn=eval_dataset_fn,
                        inputs=[eval_dataset, eval_split, eval_model2, eval_gt_key, eval_max],
                        outputs=[eval_report]
                    )

        return app

if __name__ == "__main__":
    create_app().launch(share=True)

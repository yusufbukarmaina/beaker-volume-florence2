# Florence-2 Beaker Liquid Volume Estimation (mL)

This project fine-tunes Microsoft Florence-2 for **numeric VQA** to estimate liquid volume (mL) from beaker images.

## Dataset Format (Hugging Face)

Required columns:
- `image` (HF Image / PIL)
- `prompt` (string) e.g. `What is the liquid volume in mL? Answer with a number only.`
- `label` (string) e.g. `32.5` (number-only recommended)

Recommended extra columns (for analysis):
- `volume_ml` (float)
- `beaker_ml`, `view`, `condition`, `image_name`

Splits required:
- `train` (or `train_shard_*`)
- `val`

## Run (JarvisLab)

```bash
git clone https://github.com/ictBioRtc/finetune_florence2_vision_language_model.git
cd finetune_florence2_vision_language_model
pip install -r requirements.txt
python app.py

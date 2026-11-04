"""Gradio demo: upload a brain MRI slice, get the predicted class and a Grad-CAM heatmap.

Run locally:   python app/app.py
Hugging Face Spaces: copy app/app.py, app/requirements.txt, src/ and your checkpoint into a Space
(see the README).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.gradcam import Explainer, load_checkpoint  # noqa: E402

CKPT = Path(os.environ.get("MODEL_CKPT", ROOT / "checkpoints" / "best.pt"))
if not CKPT.exists():
    raise SystemExit(f"Checkpoint not found: {CKPT}. Train a model, then copy the best one to checkpoints/best.pt "
                     "or set MODEL_CKPT.")

model, ckpt = load_checkpoint(CKPT)
explainer = Explainer(model, ckpt["timm_name"], ckpt["classes"], ckpt["img_size"])
LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary tumor"}


def run(image):
    if image is None:
        return None, None
    probs, overlay = explainer(image)
    return {LABELS.get(k, k): v for k, v in probs.items()}, overlay


examples_dir = ROOT / "app" / "examples"
examples = [[str(p)] for p in sorted(examples_dir.glob("*.jpg"))] if examples_dir.exists() else None

with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown(
        f"## Brain tumor MRI classifier\n"
        f"Model: **{ckpt['model']}**. Upload an axial brain MRI slice to see the predicted class and the regions "
        "that drove the prediction (Grad-CAM).\n\n"
        "**Research and education only. Not a medical device and not for clinical use.**"
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="MRI slice")
        with gr.Column():
            label = gr.Label(num_top_classes=4, label="Prediction")
            heat = gr.Image(label="Grad-CAM")
    gr.Button("Analyze", variant="primary").click(run, inp, [label, heat])
    if examples:
        gr.Examples(examples, inp)

if __name__ == "__main__":
    demo.launch()

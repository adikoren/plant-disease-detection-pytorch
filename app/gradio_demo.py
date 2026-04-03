"""
app/gradio_demo.py — Gradio user interface for LeafScan.

WHY Gradio instead of a custom HTML/JS frontend:
Gradio gives us a clean, interactive UI with zero frontend code. For an
ML portfolio, what matters is the model — Gradio proves it works visually
without spending days on CSS. The app is mountable inside FastAPI so one
server serves both the REST API and the UI.

WHY build_gradio_app() returns the interface object:
main.py mounts it via gr.mount_gradio_app(). Returning the object (rather than
calling .launch() here) keeps the Gradio app under FastAPI's lifecycle control.
"""

import os
import sys
from typing import Tuple

import gradio as gr
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.inference import load_model, predict
from src.utils import get_device
from torchvision import datasets


def _load_resources():
    """
    Load model and class names once at module import time.

    WHY cache at module level: when Gradio is mounted into FastAPI, this module
    is imported once. Caching avoids reloading the model on every prediction call.
    """
    device      = get_device()
    class_names = datasets.ImageFolder(config.TRAIN_DIR).classes if os.path.exists(config.TRAIN_DIR) else []
    try:
        model = load_model(config.BEST_MODEL_PATH, num_classes=config.NUM_CLASSES, device=device)
    except FileNotFoundError:
        model = None
    return model, class_names, device


_model, _class_names, _device = _load_resources()


def predict_gradio(image) -> Tuple[dict, pd.DataFrame]:
    """
    Gradio callback: receives a PIL Image, runs inference, formats output.

    WHY two outputs (Label + DataFrame):
    - gr.Label: renders a confidence bar chart for the top prediction — visually clear.
    - gr.DataFrame: shows the full top-3 table with exact percentages — precise.

    Args:
        image: PIL Image provided by the Gradio image component.

    Returns:
        Tuple of (label_dict_for_gr_Label, top3_dataframe).
    """
    if image is None:
        return {}, pd.DataFrame()

    if _model is None:
        return {"Error: Model not trained yet": 1.0}, pd.DataFrame(
            [{"Disease": "N/A", "Confidence": "N/A"}]
        )

    result = predict(_model, image, _class_names, _device)

    if not result["success"]:
        return {f"⚠️ {result['error']}": 1.0}, pd.DataFrame(
            [{"Disease": "Low confidence", "Confidence": "—"}]
        )

    # gr.Label expects {class_name: probability, ...}
    label_dict = {item["disease"]: item["confidence"] for item in result["top_3"]}

    # DataFrame for the detailed table
    df = pd.DataFrame([
        {
            "Rank":       f"#{i+1}",
            "Disease":    item["disease"].replace("___", " — ").replace("_", " "),
            "Confidence": f"{item['confidence']:.1%}",
        }
        for i, item in enumerate(result["top_3"])
    ])

    return label_dict, df


def build_gradio_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks interface.

    WHY Blocks instead of gr.Interface:
    Blocks gives layout control — we can arrange the image upload, diagnosis,
    and table side-by-side for a more professional presentation.

    Returns:
        gr.Blocks instance (not launched — caller mounts it into FastAPI).
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="LeafScan") as demo:
        gr.Markdown(
            """
            # 🌿 LeafScan — AI Plant Disease Detector
            Upload a clear, close-up photo of a plant leaf.  
            The model identifies **38 crop conditions** across 14 plant species.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Leaf Photo")
                submit_btn  = gr.Button("Diagnose 🔍", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(label="Diagnosis", num_top_classes=3)
                table_output = gr.Dataframe(label="Top 3 Predictions", headers=["Rank", "Disease", "Confidence"])

        submit_btn.click(
            fn      = predict_gradio,
            inputs  = [image_input],
            outputs = [label_output, table_output],
        )

        gr.Markdown(
            "_Confidence below 30% will be rejected. For best results, use a clean, well-lit leaf photo._"
        )

    return demo


# Allow running Gradio standalone during development
if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(server_name=config.APP_HOST, server_port=7860)

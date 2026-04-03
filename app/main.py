"""
app/main.py — FastAPI backend for LeafScan with Gradio mounted inside.

WHY FastAPI over Flask or plain Gradio:
- Async endpoints handle concurrent requests without blocking threads.
- Pydantic validation is built in — no manual input checking.
- Auto-generated /docs (Swagger UI) is essential for portfolio demos.
- Gradio is mounted AT the root so the same server serves the friendly UI
  (localhost:8000/) AND the REST API (localhost:8000/predict).

WHY lifespan for model loading:
Loading the model on startup (not per-request) is critical for performance.
ResNet50 weights are ~100MB — loading them on every /predict call would
add ~2s latency per request and crash under concurrent load.
The lifespan context manager guarantees one load at startup, one cleanup at shutdown.
"""

import io
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from app.schemas import HealthResponse, PredictionResponse, Top3Prediction
from src.inference import load_model, predict
from src.utils import get_device, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — replaces deprecated on_event('startup').

    WHY: Code before `yield` runs once at startup; code after `yield` runs
    at shutdown. FastAPI guarantees this even on crashes, preventing resource leaks.
    """
    setup_logging(config.LOG_FILE)
    logging.info("LeafScan API starting up...")

    device = get_device()

    try:
        model = load_model(config.BEST_MODEL_PATH, num_classes=config.NUM_CLASSES, device=device)
        # Store on app.state so all endpoints can access without global variables
        app.state.model       = model
        app.state.device      = device
        # WHY hardcode class_names from dataset on startup: avoids reloading the
        # full ImageFolder (slow) on every request. Classes are fixed after training.
        from src.dataset import get_transforms
        from torchvision import datasets
        app.state.class_names = datasets.ImageFolder(config.TRAIN_DIR).classes
        logging.info(f"Model loaded successfully. {len(app.state.class_names)} classes.")
    except FileNotFoundError:
        logging.warning(
            "No model checkpoint found. "
            "Run src/train.py first. /predict will return errors until model is ready."
        )
        app.state.model       = None
        app.state.device      = device
        app.state.class_names = []

    yield  # Server is running here

    logging.info("LeafScan API shutting down.")


# Mount Gradio AFTER creating the FastAPI app so its routes don't conflict
app = FastAPI(
    title=config.APP_NAME,
    description="AI-powered plant disease detection API. Upload a leaf image → get a diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Gradio UI at root "/" so browsing to localhost:8000 shows the friendly interface
try:
    from app.gradio_demo import build_gradio_app
    import gradio as gr
    gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
    logging.info("Gradio UI mounted at /ui")
except Exception as exc:
    logging.warning(f"Gradio could not be mounted: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check() -> HealthResponse:
    """
    Simple liveness check.

    WHY: Load balancers and deployment platforms (e.g. Render, Railway) poll
    /health to confirm the service is alive before routing traffic.
    """
    return HealthResponse(status="ok", app=config.APP_NAME)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_disease(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accept an uploaded leaf image and return a disease diagnosis.

    WHY async: FastAPI can handle other requests while this one waits for I/O
    (reading the uploaded file). Blocking def would stall the event loop.

    Args:
        file: Multipart image upload (JPEG, PNG, WebP, etc.).

    Returns:
        PredictionResponse — success with top-3 predictions, or error message.
    """
    # --- Guard: model must be loaded ---
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py to generate a checkpoint first.",
        )

    # --- Read and validate uploaded file ---
    try:
        image_bytes = await file.read()
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    # --- Run inference ---
    logging.info(f"Prediction request → file='{file.filename}' size={len(image_bytes)} bytes")
    result = predict(
        model       = app.state.model,
        image       = pil_image,
        class_names = app.state.class_names,
        device      = app.state.device,
    )

    # --- Build typed response ---
    if not result["success"]:
        return PredictionResponse(success=False, error=result["error"])

    top_3 = [Top3Prediction(**item) for item in result["top_3"]]
    return PredictionResponse(
        success    = True,
        disease    = result["disease"],
        confidence = result["confidence"],
        top_3      = top_3,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host   = config.APP_HOST,
        port   = config.APP_PORT,
        reload = False,  # WHY reload=False: auto-reload conflicts with lifespan model loading
    )

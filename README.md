# 🌿 LeafScan AI — Plant Disease Detection Pipeline

Welcome to **LeafScan**, a production-ready Machine Learning pipeline built from scratch in PyTorch. 

This project trains a state-of-the-art Convolutional Neural Network to correctly identify 38 different plant conditions (including healthy leaves and various diseases) from a single photo.

## 🚀 Performance
The checkpoint shipped on the `deploy/digitalocean-live` branch achieves **96.86% validation accuracy** (validation loss 0.0925), trained via transfer learning on a ResNet50 backbone with a custom training head on the **full** [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) `raw/color` split — 54,305 images across all 38 classes, stratified 85/15 into 46,159 training / 8,146 validation images with no overlap between the two. Training used the existing `src/train.py` pipeline unmodified (frozen ResNet50 backbone, Adam, ReduceLROnPlateau, early stopping) and ran the full 30 epochs configured in `config.py`; the best checkpoint (by validation accuracy and, independently, lowest validation loss) came from **epoch 28**.

## 🏗 System Architecture
This project enforces a strict Separation of Concerns, completely isolating the Neural Network calculations from the Web Server API. 

For a deep dive into the engineering decisions, transfer learning implementation, and our Out-Of-Distribution (OOD) filter that rejects non-leaf photos, please read the **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** file!

---

## 💻 Quick Start Guide

### 1. Installation
Clone the repository and install the exactly pinned dependencies in a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Web Application (Gradio + FastAPI)
The absolute easiest way to see the model in action is to launch the backend server.
```bash
python app/main.py
```
Open your browser to **http://localhost:8000/ui** to drag and drop your own leaf photos and receive instant diagnosis!

### 3. Re-Train the Model
If you've added new photos to the `data/` folder, you can fire off the automated training loop. It will automatically detect your MPS/CUDA hardware, apply Mixed Precision, and save the best checkpoint to `experiments/best_model.pth`.
```bash
python src/train.py
```

### 4. Run Evaluation Metrics
To recalculate Precision, Recall, and the 38x38 Confusion Matrix heatmap on the validation dataset:
```bash
python src/evaluate.py
```

---

## 🚢 Deployment (DigitalOcean App Platform)

The `deploy/digitalocean-live` branch is ready to deploy as-is: it ships a trained `experiments/best_model.pth` (tracked via [Git LFS](https://git-lfs.com)), a `Dockerfile`, and a `.do/app.yaml` App Platform spec.

### Checkpoint delivery — why the Dockerfile downloads the checkpoint instead of `COPY`ing it
**DigitalOcean App Platform's own git fetch of this repo does not resolve Git LFS.** If the Dockerfile just did `COPY experiments/ experiments/`, the image would silently end up with the ~130-byte LFS *pointer* text instead of the real ~108MB weights — `torch.load()` on that pointer fails with a cryptic `UnpicklingError`, which used to crash the app on startup before it could bind its port (health checks then fail with "connection refused" and the deploy is killed).

The fix: the `Dockerfile` downloads `best_model.pth` directly over plain HTTPS from **GitHub's LFS media endpoint** (`media.githubusercontent.com`), which serves the real LFS object content regardless of whether the fetching tool understands LFS — no DO-specific config, no extra storage account, no secrets, and it stays in sync automatically since it pulls from whatever's currently pushed to `deploy/digitalocean-live`:
```
https://media.githubusercontent.com/media/adikoren/plant-disease-detection-pytorch/deploy/digitalocean-live/experiments/best_model.pth
```
Right after downloading, the build calls `validate_checkpoint_file()` (`src/utils.py`) and **fails the build loudly** if the file is missing, under 10MB (an LFS pointer is ~130 bytes; a real checkpoint is 100MB+), or still literally starts with the LFS pointer header — so a broken checkpoint is caught at build time, not as a silent runtime crash. The same check also runs at app startup (`app/main.py`'s lifespan), and any load failure now logs the full traceback and serves in a degraded (no-predictions) state instead of crashing the ASGI process outright.

Class names are similarly decoupled from the dataset: `experiments/class_names.json` (small, plain git, not LFS) is the deployable source of truth for the 38 class labels, since `data/` (the training set) is intentionally never shipped to production.

If you ever need to point at a different checkpoint, override the build arg: `docker build --build-arg CHECKPOINT_URL=<url> -t leafscan .` (and set the same in `.do/app.yaml` if deploying that way).

### Deploy
1. In the DigitalOcean control panel: **Create → Apps → GitHub → adikoren/plant-disease-detection-pytorch**, branch `deploy/digitalocean-live`. App Platform detects the `Dockerfile` automatically.
   - Or from the CLI: `doctl apps create --spec .do/app.yaml`.
2. First deploy takes a few minutes (CPU-only PyTorch install, downloading the checkpoint, and baking in ImageNet weights at build time). Once live, the health check hits `/health`, and the app is served at `/` (Gradio UI is mounted at `/ui`, the REST API at `/predict`, Swagger docs at `/docs`).

### Run locally with Docker
```bash
docker build -t leafscan .
docker run -p 8000:8000 leafscan
# open http://localhost:8000/ui
```
(Docker fetches the real checkpoint itself at build time — no Git LFS setup needed just to build/run the image. You only need `git lfs install` locally if you want `experiments/best_model.pth` to contain the real weights after a plain `git clone`, e.g. to run `python app/main.py` outside Docker.)

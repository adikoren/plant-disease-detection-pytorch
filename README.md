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

The `deploy/digitalocean-live` branch is ready to deploy as-is: it ships a trained `experiments/best_model.pth` (tracked via [Git LFS](https://git-lfs.com), see below), a `Dockerfile`, and a `.do/app.yaml` App Platform spec.

### Deploy
1. Install the Git LFS filter once locally if you plan to clone/push this branch: `git lfs install`.
2. In the DigitalOcean control panel: **Create → Apps → GitHub → adikoren/plant-disease-detection-pytorch**, branch `deploy/digitalocean-live`. App Platform detects the `Dockerfile` automatically.
   - Or from the CLI: `doctl apps create --spec .do/app.yaml`.
3. First deploy takes a few minutes (CPU-only PyTorch install + baking in ImageNet weights at build time). Once live, the health check hits `/health`, and the app is served at `/` (Gradio UI is mounted at `/ui`, the REST API at `/predict`, Swagger docs at `/docs`).

### Run locally with Docker
```bash
docker build -t leafscan .
docker run -p 8000:8000 leafscan
# open http://localhost:8000/ui
```

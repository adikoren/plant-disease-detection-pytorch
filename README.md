# 🌿 LeafScan AI — Plant Disease Detection Pipeline

Welcome to **LeafScan**, a production-ready Machine Learning pipeline built from scratch in PyTorch. 

This project trains a state-of-the-art Convolutional Neural Network to correctly identify 38 different plant conditions (including healthy leaves and various diseases) from a single photo.

## 🚀 Performance
The model achieves **96.8% validation accuracy** through rigorous transfer learning, utilizing a ResNet50 backbone with a custom training head, Automatic Mixed Precision (AMP), and dynamic learning rate scheduling.

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

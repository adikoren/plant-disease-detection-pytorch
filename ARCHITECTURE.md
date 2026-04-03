# LeafScan — System Architecture & Design

This document outlines the technical architecture, design decisions, and modular structure of the LeafScan ML pipeline. It is organized to explain not just *how* the system is built, but *why* these decisions were made for a scalable, production-ready environment.

---

## 1. Directory Structure

The repository enforces a strict "Separation of Concerns" (SoC) architecture. Code is split between Machine Learning logic, API serving, and Configuration.

```text
plant-disease-detection-pytorch/
├── data/               ← Raw image datasets (train/valid/test)
├── experiments/        ← Ignored by git. Holds checkpoints (.pth), logs, and plots
├── notebooks/          ← Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                ← Core Machine Learning Pipeline
│   ├── dataset.py      ← Data loading and image transformations
│   ├── model.py        ← Neural network architecture definition
│   ├── train.py        ← Training loop, mixed precision, and schedulers
│   ├── evaluate.py     ← Offline calculation of confusion matrix and metrics
│   └── inference.py    ← Single-image prediction engine + OOD Detection
├── app/                ← Backend Server & Frontend UI
│   ├── main.py         ← FastAPI server definition and model mounting
│   ├── schemas.py      ← Pydantic schemas for data validation
│   └── gradio_demo.py  ← Gradio graphical interface
├── config.py           ← Single Source of Truth for all hyper-parameters
└── requirements.txt    ← Pinned dependency list
```

---

## 2. Core Components & Design Decisions

### **Configuration (`config.py`)**
**Decision:** Store all hyper-parameters, file paths, and constants in a single python file.
**Why:** Prevents "magic numbers" from being scattered across multiple files. If the number of classes changes from 38 to 39, we only update `config.py` in one place.

### **The Neural Network (`src/model.py`)**
**Decision:** Transfer Learning using a pre-trained **ResNet50** backbone, with a custom 3-layer Sequential head (Linear -> ReLU -> Dropout -> Linear).
**Why:** 
*   ResNet50 provides excellent feature extraction because it was pre-trained on ImageNet.
*   We freeze the backbone parameters to prevent destroying its foundational knowledge.
*   We use a custom head with **Dropout (p=0.4)** to prevent over-fitting on our specific dataset.

### **The Training Loop (`src/train.py`)**
**Decision:** Implementation of advanced PyTorch training mechanics.
**Why:** 
*   **Automatic Mixed Precision (AMP):** Uses `torch.amp` to perform calculations in FP16 instead of FP32. This drastically reduces GPU memory usage and speeds up training without losing accuracy.
*   **ReduceLROnPlateau Scheduler:** Automatically reduces the learning rate if the validation loss stops improving, ensuring the model can "fine-tune" its way into the global minimum.
*   **Early Stopping:** Stops training if the model fails to improve for 5 epochs, preventing massive overnight compute waste and overfitting.

### **Out-Of-Distribution (OOD) Detection (`src/inference.py`)**
**Decision:** A multi-stage inference pipeline utilizing a secondary `MobileNetV3` model.
**Why:** A standard ResNet trained on 38 leaf diseases is a "closed-world" model. If fed a picture of a car or a dog, it will confidently misclassify it as a leaf disease because its softmax probabilities must sum to 100%. 
To combat this, requests are first processed by `MobileNetV3` (pre-trained on 1000 general ImageNet classes). If it detects an everyday object (animal, car, person), the pipeline instantly rejects the image *before* the LeafScan model ever runs.

### **Backend Server (`app/main.py`)**
**Decision:** Use `FastAPI` with asynchronous lifespan events.
**Why:** 
*   FastAPI is incredibly fast and standard for Python ML serving.
*   The model weights (`~100MB`) are loaded into GPU memory exactly **once** during startup via a `@asynccontextmanager`, effectively making inference zero-latency on a per-request basis.

### **User Interface (`app/gradio_demo.py`)**
**Decision:** Mount a `Gradio` app directly into the FastAPI server.
**Why:** Keeps the project fully localized to Python. There is no need to write JavaScript, HTML, or CSS. Mounting Gradio within FastAPI allows one single `uvicorn` command to serve both the raw programmatic API (`/predict`) and the human visual interface (`/ui`).

---

## 3. Data Flow Example (Inference)

When a user opens `/ui` and uploads an image:
1. **Validation:** FastAPI receives the byte-stream via Gradio.
2. **Pre-Filter:** The image is passed to `OODDetector`. If it's a car/dog, it returns an error string immediately.
3. **Pre-Processing:** The image is normalized to exactly match the ImageNet statistical mean/stdev and resized to 224x224.
4. **Forward Pass:** The 4D tensor is pushed through ResNet50 in `torch.no_grad()` mode to generate logits.
5. **Post-Processing:** Logits are passed through a Softmax layer to acquire a probability distribution. The Top-3 probabilities are extracted.
6. **Thresholding:** If the top probability is below 30% (`CONFIDENCE_THRESHOLD`), the model refuses to diagnose. Otherwise, the JSON payload is formatted and returned to the browser.

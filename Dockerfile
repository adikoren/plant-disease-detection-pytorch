# Dockerfile — LeafScan production image for DigitalOcean App Platform.
#
# WHY CPU-only torch wheels: the App Platform tier this app targets has no GPU.
# The default PyPI torch wheel bundles CUDA and is ~4x larger; pulling the CPU
# build from PyTorch's own index keeps the image small and the build fast.
FROM python:3.11-slim

WORKDIR /srv/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY config.py .
COPY app/ app/
COPY src/ src/
COPY experiments/class_names.json experiments/class_names.json

# WHY download the checkpoint instead of `COPY experiments/`: DigitalOcean App
# Platform's own git fetch of this repo does not resolve Git LFS, so a plain
# COPY would silently ship the ~130-byte LFS *pointer* text instead of the real
# ~108MB weights (that's exactly what caused the app to crash on startup with
# a bare UnpicklingError and never bind its port). GitHub serves the real LFS
# object content over plain HTTPS via media.githubusercontent.com regardless
# of whether the fetching tool understands LFS — see "Checkpoint delivery" in
# README.md. validate_checkpoint_file() then fails the BUILD loudly (not the
# deploy) if what we got is missing, truncated, or still an LFS pointer.
ARG CHECKPOINT_URL=https://media.githubusercontent.com/media/adikoren/plant-disease-detection-pytorch/deploy/digitalocean-live/experiments/best_model.pth
RUN python -c "import urllib.request; urllib.request.urlretrieve('${CHECKPOINT_URL}', 'experiments/best_model.pth')" \
    && python -c "from src.utils import validate_checkpoint_file; validate_checkpoint_file('experiments/best_model.pth')" \
    && echo "Checkpoint OK: $(stat -c%s experiments/best_model.pth 2>/dev/null || stat -f%z experiments/best_model.pth) bytes"

# WHY pre-fetch here: model.py and inference.py's OOD detector both construct
# torchvision models with pretrained ImageNet weights. Baking those weights
# into the image means the container never needs internet access at runtime
# to serve a first request.
RUN python -c "from torchvision import models; \
    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1); \
    models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

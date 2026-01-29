# Python light version
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    "pillow>=11.0.0"

# Pytorch
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN useradd -m appuser
USER appuser

# Copy script and model weights
COPY src/api.py .
COPY --chown=appuser:appuser pipeline3/effnet_s/finetuned/model.pth ./weights/model.pth

EXPOSE 8080
CMD ["python", "api.py"]
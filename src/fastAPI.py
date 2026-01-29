import time
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms, models
from pydantic import BaseModel
from typing import Dict

# --- Configurazione Percorsi e Costanti ---
MODEL_PATH = "../baseline/model.pth"
MODEL_VERSION = "convnext_tiny"  # Aggiornato in base ai pesi effettivi
CLASSES = [
    "demodicosis",
    "dermatitis",
    "fungal_infections",
    "healthy",
    "hypersensitivity",
    "ringworm"
]

app = FastAPI(title="Dog Skin Disease Classifier - ConvNeXt Tiny")


# --- Inizializzazione Modello ---
def load_model():
    # Cambiato da base a tiny per risolvere il mismatch dei canali (96 vs 128)
    model = models.convnext_tiny(weights=None)

    # In ConvNeXt il classifier è model.classifier[2]
    n_inputs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(n_inputs, len(CLASSES))

    # Caricamento dello state_dict
    device = torch.device("cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Carichiamo i pesi
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# Caricamento del modello all'avvio
model, device = load_model()

# Preprocessing: Solo resize a 224x224 (No normalizzazione)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --- Schemi di Output ---
class PredictionResponse(BaseModel):
    label_name: str
    confidence_score: float
    all_probabilities: Dict[str, float]
    inference_time_ms: float
    model_version_id: str


# --- Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    conf, idx = torch.max(probabilities, 0)
    label = CLASSES[idx.item()]

    prob_dist = {CLASSES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASSES))}

    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000

    return {
        "label_name": label,
        "confidence_score": round(float(conf), 4),
        "all_probabilities": prob_dist,
        "inference_time_ms": round(inference_time, 2),
        "model_version_id": MODEL_VERSION
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
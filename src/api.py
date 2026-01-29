import io
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

MODEL_PATH = "weights/model.pth"  # Container path
MODEL_VERSION = "efficientnet_v2_s"

CLASSES = [
    "demodicosis",
    "dermatitis",
    "fungal_infections",
    "healthy",
    "hypersensitivity",
    "ringworm",
]

app = FastAPI(title="Dog Skin Disease Classifier")


def load_model() -> Tuple[nn.Module, torch.device]:
    """Load the EfficientNetV2_S model and its fine-tuned weights.

    The architecture is modified by replacing the final classifier
    to adapt it to the problem-specific number of classes.

    Returns:
        Tuple[nn.Module, torch.device]: The loaded model and the device (CPU).

    """
    # Initialize the model
    model = getattr(models, MODEL_VERSION)(weights=None)

    # In EfficientNetV2, classifier is accessible through model.classifier[1]
    # Structure: [0] Dropout, [1] Linear
    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, len(CLASSES))

    # state_dict loading
    device = torch.device("cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# Model is loaded when the application starts
model, device = load_model()

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PredictionResponse(BaseModel):
    """Response pattern for model prediction.

    Attributes:
        label_name (str): Name of the predicted class.
        confidence_score (float): Probability associated with the predicted class.
        inference_time_ms (float): Time taken for inference in milliseconds.
        model_version_id (str): Model version identifier.

    """

    label_name: str
    confidence_score: float
    inference_time_ms: float
    model_version_id: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> Dict:
    """Upon receiving an image, it performs preprocessing and returns the prediction.

    Args:
        file (UploadFile): Image file uploaded via POST request.

    Returns:
        Dict: Classification result with score and execution time.

    """
    start_time = time.perf_counter()

    # Read the uploaded image
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    # Prepare the image to be fed as input to the model
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Inference without gradient calculation
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs[0], dim=0)

    # Class and confidence score extraction
    conf, idx = torch.max(probabilities, 0)
    label = CLASSES[idx.item()]

    return {
        "label_name": label,
        "confidence_score": round(float(conf), 4),
        "inference_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "model_version_id": MODEL_VERSION,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

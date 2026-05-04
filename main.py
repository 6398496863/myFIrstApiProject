"""
Pneumonia Detection API - FastAPI Backend

Uses a ViT model fine-tuned on chest X-rays
from Hugging Face Hub.

Model:
nickmuchi/vit-finetuned-chest-xray-pneumonia
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import torch
import time
from typing import List, Dict
import logging
from transformers import pipeline

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- App --------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="Deep learning-based chest X-ray analysis",
    version="1.0.0"
)

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Model --------------------
MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
classifier = None


@app.on_event("startup")
async def load_model():
    global classifier
    logger.info(f"Loading model: {MODEL_ID}")

    try:
        classifier = pipeline("image-classification", model=MODEL_ID)
        logger.info("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")


# -------------------- Schemas --------------------
class ScoreItem(BaseModel):
    label: str
    score: float


class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: List[ScoreItem]
    inference_time_ms: float
    model_id: str
    verdict: str
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str


# -------------------- Helper --------------------
def map_risk(label: str, confidence: float):
    is_pneumonia = "PNEUMONIA" in label.upper()

    if is_pneumonia:
        if confidence >= 0.85:
            return (
                "Pneumonia detected - Please consult a physician immediately",
                "HIGH"
            )
        else:
            return (
                "Possible Pneumonia - Further evaluation is needed",
                "MEDIUM"
            )
    else:
        return (
            "Normal - No signs of pneumonia detected",
            "LOW"
        )


# -------------------- Routes --------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="OK",
        model_loaded=classifier is not None,
        model_id=MODEL_ID,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):

    # Check model
    if classifier is None:
        raise HTTPException(503, "Model not loaded yet")

    # Validate file
    if file.content_type not in [
        "image/jpeg", "image/png", "image/jpg", "image/webp"
    ]:
        raise HTTPException(400, "Only image files supported")

    # Read image
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    # -------- Inference --------
    t0 = time.perf_counter()

    try:
        results = classifier(image, top_k=None)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Sort results
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if not results:
        raise HTTPException(500, "Empty prediction result")

    top = results[0]

    verdict, risk_level = map_risk(top["label"], top["score"])

    return PredictionResult(
        label=top["label"],
        confidence=top["score"],
        all_scores=[
            ScoreItem(label=r["label"], score=r["score"])
            for r in results
        ],
        inference_time_ms=round(elapsed_ms, 2),
        model_id=MODEL_ID,
        verdict=verdict,
        risk_level=risk_level
    )


@app.get("/model.info")
async def model_info():
    return {
        "model_id": MODEL_ID,
        "architecture": "Vision Transformer (ViT)",
        "task": "Chest X-ray classification",
        "classes": ["NORMAL", "PNEUMONIA"],
        "source": "Hugging Face"
    }





    
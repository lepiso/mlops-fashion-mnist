import os
import time
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput,
    HealthResponse, ModelInfoResponse, FASHION_LABELS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
feature_names = None
model_loaded_at = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names, model_loaded_at
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load("models/feature_names.pkl")
        model_loaded_at = time.time()
        logger.info(f"Modele charge : {model_path}")
    except FileNotFoundError:
        logger.warning("Modele non trouve — lancez train.py d'abord")
    yield

app = FastAPI(
    title="Fashion MNIST API",
    description="Classification de vetements — 10 categories",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Fashion MNIST Prediction API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_status="loaded" if model else "not_loaded",
        model_uptime_seconds=time.time() - model_loaded_at if model_loaded_at else 0,
        version="2.0.0"
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if not model:
        raise HTTPException(503, "Modele non charge")
    return ModelInfoResponse(
        model_type=type(model.named_steps["model"]).__name__,
        n_features=784,
        n_classes=10,
        class_labels={str(k): v for k, v in FASHION_LABELS.items()}
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    if not model:
        raise HTTPException(503, "Modele non disponible")
    try:
        X = pd.DataFrame([data.features], columns=feature_names)
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()
        top3  = sorted(
            [{"class": int(i), "label": FASHION_LABELS[i], "probability": round(proba[i], 4)}
            for i in range(10)],
            key=lambda x: x["probability"], reverse=True
        )[:3]
        return PredictionOutput(
            prediction=pred,
            label=FASHION_LABELS[pred],
            probabilities=[round(p, 4) for p in proba],
            confidence=round(max(proba), 4),
            top3=top3
        )
    except Exception as e:
        raise HTTPException(422, str(e))

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchPredictionInput):
    if not model:
        raise HTTPException(503, "Modele non disponible")
    results = []
    for item in batch.inputs:
        X     = pd.DataFrame([item.features], columns=feature_names)
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()
        top3  = sorted(
            [{"class": int(i), "label": FASHION_LABELS[i], "probability": round(proba[i], 4)}
            for i in range(10)],
            key=lambda x: x["probability"], reverse=True
        )[:3]
        results.append(PredictionOutput(
            prediction=pred, label=FASHION_LABELS[pred],
            probabilities=[round(p, 4) for p in proba],
            confidence=round(max(proba), 4), top3=top3
        ))
    return BatchPredictionOutput(predictions=results, total=len(results))

@app.get("/predict/example")
async def example():
    pixels = np.random.uniform(0, 1, 784).round(4).tolist()
    return {
        "note": "Envoyez 784 valeurs de pixels entre 0.0 et 1.0",
        "example_request": {"features": pixels},
        "classes": FASHION_LABELS
    }
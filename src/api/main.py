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

# --- VARIABLES GLOBALES ---
models = {}
feature_names = None
model_loaded_at = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, feature_names, model_loaded_at
    try:
        # On charge les nouveaux modèles
        models["rf"] = joblib.load("models/rf_model.pkl")
        models["mlp"] = joblib.load("models/mlp_model.pkl")
        
        # ASTUCE : Pour que tes tests existants ne cassent pas, 
        # on peut aussi charger rf_model.pkl en tant que 'model' générique
        models["model"] = models["rf"] 
        
        feature_names = joblib.load("models/feature_names.pkl")
        model_loaded_at = time.time()
        logger.info(f"🚀 Modèles chargés avec succès : {list(models.keys())}")
    except Exception as e:
        logger.warning(f"⚠️ Erreur chargement modèles : {e}")
    yield

app = FastAPI(
    title="Fashion MNIST Multi-Model API",
    description="Inférence via Random Forest ou MLP",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 1. ROUTE ROOT (Indispensable pour test_api_root)
@app.get("/")
async def root():
    return {"message": "Fashion MNIST Prediction API", "docs": "/docs"}

# 2. ROUTE HEALTH
@app.get("/health", response_model=HealthResponse)
async def health():
    is_ready = "rf" in models
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_status="loaded" if is_ready else "not_loaded",
        model_uptime_seconds=time.time() - model_loaded_at if model_loaded_at else 0,
        version="2.1.0"
    )

# 3. ROUTE PREDICT (Unique)
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    model_key = data.model_type
    if model_key not in models:
        model_key = "rf" # Fallback sécurisé

    try:
        current_model = models[model_key]
        # On utilise numpy pour éviter le warning de StandardScaler
        X = np.array(data.features).reshape(1, -1)
        
        pred = int(current_model.predict(X)[0])
        proba = current_model.predict_proba(X)[0].tolist()
        
        top3 = sorted(
            [{"label": FASHION_LABELS[i], "probability": round(proba[i], 4)}
            for i in range(10)],
            key=lambda x: x["probability"], reverse=True
        )[:3]
        
        return PredictionOutput(
            prediction=pred,
            label=FASHION_LABELS[pred],
            probabilities=[round(p, 4) for p in proba],
            confidence=round(max(proba), 4),
            top3=top3,
            model_used=model_key
        )
    except Exception as e:
        raise HTTPException(422, str(e))

# 4. ROUTE BATCH (Indispensable pour test_api_predict_batch)
@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchPredictionInput):
    results = []
    for item in batch.inputs:
        res = await predict(item)
        results.append(res)
    return BatchPredictionOutput(predictions=results, total=len(results))

# 5. ROUTE MODEL INFO (Indispensable pour test_api_model_info)
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if "rf" not in models:
        raise HTTPException(503, "Modèles non disponibles")
    return ModelInfoResponse(
        model_type="Multi-Model (RF/MLP)",
        n_features=784,
        n_classes=10,
        class_labels={str(k): v for k, v in FASHION_LABELS.items()}
    )

# 6. ROUTE EXAMPLE (Indispensable pour test_api_example)
@app.get("/predict/example")
async def example():
    return {
        "example_request": {"features": [0.0]*784, "model_type": "rf"},
        "classes": FASHION_LABELS
    }
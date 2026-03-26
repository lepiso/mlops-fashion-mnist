import os
import time
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Assure-toi que schemas.py contient bien le champ model_used dans PredictionOutput
from .schemas import (
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput,
    HealthResponse, ModelInfoResponse, FASHION_LABELS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- VARIABLES GLOBALES ---
models = {}  # Dictionnaire pour stocker RF et MLP
feature_names = None
model_loaded_at = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, feature_names, model_loaded_at
    try:
        # Chargement des deux modèles
        models["rf"] = joblib.load("models/rf_model.pkl")
        models["mlp"] = joblib.load("models/mlp_model.pkl")
        
        # Chargement des noms de colonnes (pour le DataFrame)
        feature_names = joblib.load("models/feature_names.pkl")
        
        model_loaded_at = time.time()
        logger.info(f"Modèles chargés : RF et MLP")
    except FileNotFoundError as e:
        logger.warning(f"Fichier manquant : {e}. Vérifiez le dossier models/")
    yield

app = FastAPI(
    title="Fashion MNIST Multi-Model API",
    description="Inférence via Random Forest ou MLP Neural Network",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health", response_model=HealthResponse)
async def health():
    is_ready = "rf" in models and "mlp" in models
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_status=f"Loaded: {list(models.keys())}" if is_ready else "incomplete",
        model_uptime_seconds=time.time() - model_loaded_at if model_loaded_at else 0,
        version="2.1.0"
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    # 1. Vérification de la disponibilité du modèle demandé
    model_key = data.model_type # Utilise le champ ajouté dans schemas.py
    if model_key not in models:
        raise HTTPException(503, f"Le modèle '{model_key}' n'est pas chargé")

    try:
        current_model = models[model_key]
        
        # 2. Préparation des données (DataFrame pour garder les noms de features)
        X = pd.DataFrame([data.features], columns=feature_names)
        
        # 3. Inférence
        pred  = int(current_model.predict(X)[0])
        proba = current_model.predict_proba(X)[0].tolist()
        
        # 4. Calcul du Top 3
        top3  = sorted(
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
            model_used=model_key # Pour confirmer à Streamlit quel modèle a répondu
        )
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(422, str(e))

# ... (Tu peux garder les autres routes comme batch en les adaptant de la même façon)
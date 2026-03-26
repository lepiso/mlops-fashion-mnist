# --- Section Modèle ---
import os

import numpy as np


def test_model_exists():
    """Vérifie que les DEUX modèles existent"""
    assert os.path.exists("models/rf_model.pkl"), "rf_model.pkl manquant"
    assert os.path.exists("models/mlp_model.pkl"), "mlp_model.pkl manquant"

def test_model_loads():
    """Charge le modèle par défaut (RF)"""
    import joblib
    model = joblib.load("models/rf_model.pkl")
    assert model is not None

# --- Section API ---

def test_api_predict_valid(client):
    """Correction : On ajoute explicitement le model_type"""
    pixels = np.random.uniform(0, 1, 784).round(4).tolist()
    payload = {
        "features": pixels,
        "model_type": "rf" 
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 422:
        print(f"DEBUG 422: {response.json()}")
        
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "model_used" in response.json()

def test_api_predict_batch(client):
    """Correction : Chaque élément du batch doit avoir son model_type"""
    pixels = np.random.uniform(0, 1, 784).round(4).tolist()
    batch = {
        "inputs": [
            {"features": pixels, "model_type": "rf"},
            {"features": pixels, "model_type": "mlp"}
        ]
    }
    response = client.post("/predict/batch", json=batch)
    assert response.status_code in [200, 503]
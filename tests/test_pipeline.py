"""
Tests automatiques du pipeline MLOps Fashion MNIST.
Lance avec : pytest tests/ -v
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tests des données ──────────────────────────────────────────────────────

def test_dataset_exists():
    """Le dataset doit exister après generate_data.py"""
    assert os.path.exists("data/raw/dataset.csv"), \
        "dataset.csv manquant — lancez generate_data.py"

def test_dataset_shape():
    """Le dataset doit avoir 786 colonnes (784 pixels + target + label_name)"""
    import pandas as pd
    df = pd.read_csv("data/raw/dataset.csv")
    assert df.shape[1] == 786, f"Attendu 786 colonnes, obtenu {df.shape[1]}"
    assert len(df) >= 1000, f"Trop peu d'images : {len(df)}"

def test_dataset_pixels_normalized():
    """Les pixels doivent être normalisés entre 0 et 1"""
    import pandas as pd
    df = pd.read_csv("data/raw/dataset.csv")
    pixel_cols = [f"pixel_{i}" for i in range(784)]
    assert df[pixel_cols].min().min() >= 0.0, "Pixels < 0 détectés"
    assert df[pixel_cols].max().max() <= 1.0, "Pixels > 1 détectés"

def test_dataset_classes():
    """Le dataset doit avoir exactement 10 classes (0 à 9)"""
    import pandas as pd
    df = pd.read_csv("data/raw/dataset.csv")
    classes = sorted(df["target"].unique())
    assert classes == list(range(10)), f"Classes incorrectes : {classes}"

def test_no_missing_values():
    """Pas de valeurs manquantes dans le dataset"""
    import pandas as pd
    df = pd.read_csv("data/raw/dataset.csv")
    assert df.isnull().sum().sum() == 0, "Valeurs manquantes détectées"

# ── Tests du modèle ────────────────────────────────────────────────────────

def test_model_exists():
    """Le modèle doit exister après train.py"""
    assert os.path.exists("models/model.pkl"), \
        "model.pkl manquant — lancez train.py"

def test_feature_names_exist():
    """Les noms des features doivent exister"""
    assert os.path.exists("models/feature_names.pkl"), \
        "feature_names.pkl manquant"

def test_model_loads():
    """Le modèle doit se charger sans erreur"""
    import joblib
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_model_predicts():
    """Le modèle doit faire des prédictions correctes"""
    import joblib
    import pandas as pd
    model = joblib.load("models/model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    # Image noire (784 zéros)
    X = pd.DataFrame([np.zeros(784)], columns=feature_names)
    pred = model.predict(X)
    assert pred[0] in range(10), f"Prédiction invalide : {pred[0]}"

def test_model_probabilities():
    """Les probabilités doivent sommer à 1"""
    import joblib
    import pandas as pd
    model = joblib.load("models/model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    X = pd.DataFrame([np.random.uniform(0, 1, 784)], columns=feature_names)
    proba = model.predict_proba(X)[0]
    assert len(proba) == 10, "Doit avoir 10 probabilités"
    assert abs(sum(proba) - 1.0) < 1e-5, "Les probabilités ne somment pas à 1"

def test_model_accuracy():
    """Le modèle doit avoir au moins 80% d'accuracy"""
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    model = joblib.load("models/model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    df = pd.read_csv("data/raw/dataset.csv")
    if "label_name" in df.columns:
        df = df.drop(columns=["label_name"])
    X = df.drop(columns=["target"])
    y = df["target"]
    # Test sur 500 exemples pour être rapide
    X_sample = X.sample(500, random_state=42)
    y_sample = y[X_sample.index]
    acc = accuracy_score(y_sample, model.predict(X_sample))
    assert acc >= 0.80, f"Accuracy trop faible : {acc:.2%} (minimum 80%)"

# ── Tests des graphes ──────────────────────────────────────────────────────

def test_reports_exist():
    """Les graphes doivent exister après train.py"""
    expected = [
        "reports/confusion_matrix_RandomForest.png",
        "reports/confusion_matrix_MLP_NeuralNet.png",
        "reports/models_comparison.png",
        "reports/class_distribution.png",
    ]
    for path in expected:
        assert os.path.exists(path), f"Graphe manquant : {path}"

def test_metrics_json_exists():
    """Le fichier de métriques doit exister"""
    assert os.path.exists("reports/metrics.json"), "metrics.json manquant"

def test_metrics_json_content():
    """Les métriques doivent avoir accuracy > 0.80"""
    import json
    with open("reports/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics, "accuracy manquant dans metrics.json"
    assert metrics["accuracy"] >= 0.80, \
        f"Accuracy trop faible : {metrics['accuracy']}"

# ── Tests de l'API ─────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Crée un client de test FastAPI"""
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)

def test_api_root(client):
    """GET / doit retourner 200"""
    response = client.get("/")
    assert response.status_code == 200

def test_api_health(client):
    """GET /health doit retourner status healthy ou degraded"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data

def test_api_predict_valid(client):
    """POST /predict avec 784 pixels valides doit retourner 200 ou 503"""
    pixels = np.random.uniform(0, 1, 784).tolist()
    response = client.post("/predict", json={"features": pixels})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "confidence" in data
        assert "top3" in data
        assert data["prediction"] in range(10)
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["top3"]) == 3

def test_api_predict_wrong_size(client):
    """POST /predict avec mauvais nombre de pixels doit retourner 422"""
    pixels = [0.5] * 100  # Seulement 100 pixels au lieu de 784
    response = client.post("/predict", json={"features": pixels})
    assert response.status_code == 422

def test_api_predict_batch(client):
    """POST /predict/batch doit gérer plusieurs images"""
    pixels = np.random.uniform(0, 1, 784).tolist()
    batch = {"inputs": [{"features": pixels}, {"features": pixels}]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

def test_api_model_info(client):
    """GET /model/info doit retourner les infos du modèle"""
    response = client.get("/model/info")
    # 200 si modèle chargé, 503 si non chargé
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["n_features"] == 784
        assert data["n_classes"] == 10

def test_api_example(client):
    """GET /predict/example doit retourner un exemple"""
    response = client.get("/predict/example")
    assert response.status_code == 200
    data = response.json()
    assert "example_request" in data
    assert len(data["example_request"]["features"]) == 784
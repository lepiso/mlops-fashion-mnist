from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

FASHION_LABELS = {
    0: "T-shirt/top", 1: "Trouser",  2: "Pullover",
    3: "Dress",       4: "Coat",     5: "Sandal",
    6: "Shirt",       7: "Sneaker",  8: "Bag",
    9: "Ankle boot"
}

class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="784 pixels normalises [0,1]")
    # NOUVEAU : Choix du modèle (optionnel, 'rf' par défaut)
    model_type: str = Field("rf", description="Type de modèle à utiliser : 'rf' ou 'mlp'")

    @field_validator("features")
    @classmethod
    def check_pixels(cls, v):
        if len(v) != 784:
            raise ValueError(f"Attendu 784 pixels, recu {len(v)}")
        return v
    
    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v):
        allowed = ["rf", "mlp"]
        if v not in allowed:
            raise ValueError(f"Modèle '{v}' non supporté. Choisissez parmi {allowed}")
        return v

class PredictionOutput(BaseModel):
    prediction: int
    label: str
    probabilities: List[float]
    confidence: float
    top3: List[dict]
    model_used: str # Ajouté pour confirmer quel modèle a répondu

# Les autres classes restent identiques car elles utilisent PredictionInput
class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total: int

class HealthResponse(BaseModel):
    status: str
    model_status: str
    model_uptime_seconds: float
    version: str

class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    n_classes: int
    class_labels: dict
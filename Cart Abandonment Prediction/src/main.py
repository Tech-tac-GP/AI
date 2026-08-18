from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import json
import os
from catboost import CatBoostClassifier
from cart_abandonment.src.preprocessing import preprocess_session_data

# Initialize router
router = APIRouter()

# Dynamically resolve the absolute path to the cart_abandonment directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metadata_path = os.path.join(BASE_DIR, "models", "metadata.json")
model_path = os.path.join(BASE_DIR, "models", "cart_abandonment_model.cbm")

# Load model and metadata using the dynamic paths
with open(metadata_path, "r") as f:
    metadata = json.load(f)

model = CatBoostClassifier()
model.load_model(model_path)

class SessionPayload(BaseModel):
    clicks: List[Dict]

class PredictionResponse(BaseModel):
    abandonment_probability: float
    predicted_action: str

@router.post("/predict", response_model=PredictionResponse)
def predict_abandonment(payload: SessionPayload):
    # Preprocess incoming session data
    df = preprocess_session_data(payload.clicks, metadata)
    
    # Predict probability of abandonment (class 1)
    probability = float(model.predict_proba(df)[:, 1][0])
    
    # Threshold check
    threshold = metadata.get("threshold", 0.50)
    if probability >= threshold:
        action = "ABANDONED (Trigger Discount!)"
    else:
        action = "CONVERTED (Do Nothing)"
        
    return {
        "abandonment_probability": round(probability, 4),
        "predicted_action": action
    }

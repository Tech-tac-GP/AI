from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import json
from catboost import CatBoostClassifier

from src.preprocessing import preprocess_session_data

app = FastAPI(title="Cart Abandonment Prediction API")

# Load model and metadata
with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

model = CatBoostClassifier()
model.load_model("models/cart_abandonment_model.cbm")

class SessionPayload(BaseModel):
    clicks: List[Dict]

class PredictionResponse(BaseModel):
    abandonment_probability: float
    predicted_action: str

@app.post("/predict", response_model=PredictionResponse)
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
# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json
from catboost import CatBoostClassifier

from src.preprocessing import preprocess_session_data 

app = FastAPI(
    title="Purchase Prediction API",
    description="Machine Learning service to predict e-commerce purchase probability.",
    version="1.0.0",
    docs_url="/"
)

# Load Model & Metadata
with open("models/metadata.json", "r") as f:
    metadata = json.load(f)
model = CatBoostClassifier().load_model("models/catboost_model.cbm")

class RawClicksPayload(BaseModel):
    clicks: List[Dict] # Expects a list of 3 click events from the frontend

@app.post("/predict")
def predict(payload: RawClicksPayload):
    try:
        # Let the Data Science script format the data
        clean_df = preprocess_session_data(payload.clicks, metadata)
        
        # Let the AI make the prediction
        probability = float(model.predict_proba(clean_df)[:, 1][0])
        is_buyer = int(probability >= metadata["optimal_threshold"])
        
        # Return the result
        return {"intent": is_buyer, "probability": probability}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

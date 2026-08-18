import os
import json
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, HTTPException
from demand_forecasting.api.schemas import ForecastRequest, ForecastResponse

router = APIRouter()

# Path Resolution 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "demand_forecast_xgboost.json")
META_PATH = os.path.join(ROOT_DIR, "models", "metadata.json")

print(f"--- SERVER STARTUP ---")
print(f"Loading model from: {MODEL_PATH}")

# Load using native XGBoost Booster (Bypasses sklearn wrapper issues)
model = xgb.Booster()
model.load_model(MODEL_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

print("Model and metadata successfully loaded into memory!")

# Prediction Endpoint
@router.post("/predict", response_model=ForecastResponse)
def predict_demand(request: ForecastRequest):
    try:
        input_dict = request.model_dump()
        feature_data = {k: v for k, v in input_dict.items() if k != "product_id"}
        
        # Create a DMatrix (XGBoost's optimized native data structure)
        input_df = pd.DataFrame([feature_data])
        expected_features = metadata.get("expected_features", list(feature_data.keys()))
        input_df = input_df[expected_features]
        
        dmatrix = xgb.DMatrix(input_df)

        # Generate prediction natively
        raw_prediction = model.predict(dmatrix)[0]
        
        final_prediction = max(0.0, float(raw_prediction))

        return ForecastResponse(
            product_id=request.product_id,
            forecasted_units=round(final_prediction, 2),
            message="Prediction generated successfully."
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

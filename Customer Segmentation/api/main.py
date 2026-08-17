from fastapi import FastAPI, HTTPException

from api.schemas import CustomerFeatures, SegmentResponse
from src.inference.predict_segment import SegmentPredictor

app = FastAPI(
    title="Customer Segmentation API",
    version="1.0.0",
    docs_url="/"
)

predictor = SegmentPredictor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=SegmentResponse)
def predict(customer: CustomerFeatures):
    try:
        return predictor.predict(customer.model_dump())

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

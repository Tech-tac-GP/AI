from fastapi import APIRouter, HTTPException
from customer_segmentation.api.schemas import CustomerFeatures, SegmentResponse
from customer_segmentation.src.inference.predict_segment import SegmentPredictor

router = APIRouter()

predictor = SegmentPredictor()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict", response_model=SegmentResponse)
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

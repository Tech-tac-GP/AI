# Customer Segmentation

Production structure for the customer segmentation project.

Flow:
raw events -> session/customer features -> Scaler -> K-Means -> segment -> FastAPI

## API
Run from the project root:

    pip install -r requirements.txt
    uvicorn api.main:app --reload

## Model artifacts
Put these in `models/`:

- customer_segmentation_kmeans.pkl
- customer_segmentation_scaler.pkl

## Important
K-Means cluster IDs do not have inherent business meaning.
Update `SEGMENT_NAMES` after reviewing your final cluster profile.

# Customer Segmentation

Production structure for the customer segmentation project.

Flow:
raw events -> session/customer features -> StandardScaler -> K-Means -> segment -> FastAPI

## API
Run from the project root:

    pip install -r requirements.txt
    uvicorn api.main:app --reload

Swagger:
http://127.0.0.1:8000/docs

## Model artifacts
Put these in `models/`:

- customer_segmentation_kmeans.pkl
- customer_segmentation_scaler.pkl

PCA is only needed at inference if K-Means was actually trained on PCA-transformed data.

## Feature contract
The API expects exactly the ten features in `src/utils/config.py`.
Their order must match the order used to fit the saved scaler.

## Important
K-Means cluster IDs do not have inherent business meaning.
Update `SEGMENT_NAMES` after reviewing your final cluster profile.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

KMEANS_PATH = MODEL_DIR / "customer_segmentation_kmeans.pkl"
SCALER_PATH = MODEL_DIR / "customer_segmentation_scaler.pkl"

# IMPORTANT:
# Replace this list with the exact 9 feature names printed by:
# scaler.feature_names_in_
#
# The saved RobustScaler confirms that the trained model expects 9 features.
SEGMENT_FEATURES = [
    "purchase_rate",
    "cart_rate",
    "view_rate",
    "events_per_session",
    "products_per_session",
    "total_spending",
    "average_purchase_value",
    "days_since_last_activity",
    "activity_days",
]

# Your final trained K-Means has 4 clusters.
# Rename these after checking your final cluster profiles.
SEGMENT_NAMES = {
    0: "Low Engagement Customers",
    1: "High Intent Customers",
    2: "High Value Customers",
    3: "Loyal Active Customers",
}

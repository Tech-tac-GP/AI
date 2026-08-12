import pandas as pd
from src.utils.config import SEGMENT_FEATURES

def select_features(df):
    if not SEGMENT_FEATURES:
        raise ValueError(
            "SEGMENT_FEATURES is empty. "
            "Run the feature inspection script and add the exact 9 "
            "feature names used during training."
        )

    missing = [c for c in SEGMENT_FEATURES if c not in df.columns]

    if missing:
        raise ValueError(f"Missing segmentation features: {missing}")

    return df[SEGMENT_FEATURES].copy()

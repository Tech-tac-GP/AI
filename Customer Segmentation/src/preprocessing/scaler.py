import joblib

from src.utils.config import SCALER_PATH
from src.preprocessing.transform import select_features

def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"RobustScaler not found: {SCALER_PATH}"
        )

    return joblib.load(SCALER_PATH)

def transform(df):
    scaler = load_scaler()
    X = select_features(df)
    return scaler.transform(X)

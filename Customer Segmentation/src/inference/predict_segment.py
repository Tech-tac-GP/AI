import joblib
import pandas as pd

from customer_segmentation.src.utils.config import (
    KMEANS_PATH,
    SCALER_PATH,
    SEGMENT_FEATURES,
    SEGMENT_NAMES,
)

class SegmentPredictor:

    def __init__(self):
        if not KMEANS_PATH.exists():
            raise FileNotFoundError(
                f"K-Means model not found: {KMEANS_PATH}"
            )

        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"RobustScaler not found: {SCALER_PATH}"
            )

        self.model = joblib.load(KMEANS_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        # Validate the saved model contract.
        scaler_features = getattr(
            self.scaler,
            "feature_names_in_",
            None
        )

        if scaler_features is not None:
            scaler_features = list(scaler_features)

            if SEGMENT_FEATURES and scaler_features != SEGMENT_FEATURES:
                raise ValueError(
                    "SEGMENT_FEATURES does not match the feature order "
                    "used when the RobustScaler was fitted.\n"
                    f"Saved scaler: {scaler_features}\n"
                    f"Configured: {SEGMENT_FEATURES}"
                )

        if getattr(self.model, "n_features_in_", None) != 9:
            raise ValueError(
                f"Expected K-Means to use 9 features, but it expects "
                f"{self.model.n_features_in_}."
            )

    def predict(self, customer):
        if not SEGMENT_FEATURES:
            raise ValueError(
                "SEGMENT_FEATURES is empty. Add the exact 9 training "
                "features to src/utils/config.py."
            )

        missing = [
            feature
            for feature in SEGMENT_FEATURES
            if feature not in customer
        ]

        if missing:
            raise ValueError(
                f"Missing customer features: {missing}"
            )

        X = pd.DataFrame(
            [[customer[feature] for feature in SEGMENT_FEATURES]],
            columns=SEGMENT_FEATURES
        )

        # IMPORTANT:
        # The trained pipeline is:
        #
        # customer features -> RobustScaler -> K-Means
        #
        # PCA is NOT used here because K-Means expects 9 features,
        # while the saved PCA produces 2 components.

        X_scaled = self.scaler.transform(X)

        cluster = int(self.model.predict(X_scaled)[0])

        return {
            "cluster": cluster,
            "segment": SEGMENT_NAMES.get(
                cluster,
                f"Cluster {cluster}"
            )
        }

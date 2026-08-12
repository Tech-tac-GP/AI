import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from src.preprocessing.transform import select_features
from src.utils.config import KMEANS_PATH, SCALER_PATH

def train_kmeans(customer_df, n_clusters=4, random_state=42):
    X = select_features(customer_df)

    # This matches the scaler type used by the saved production model.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )

    model.fit(X_scaled)

    KMEANS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, KMEANS_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

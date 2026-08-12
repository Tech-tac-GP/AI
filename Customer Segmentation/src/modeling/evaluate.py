import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_kmeans(model, X_scaled):
    labels = model.labels_
    return {
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels))
    }

def cluster_profile(customer_df, labels):
    result = customer_df.copy()
    result["cluster"] = labels
    return result.groupby("cluster").mean(numeric_only=True)

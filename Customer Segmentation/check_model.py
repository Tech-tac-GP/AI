import joblib

KMEANS_PATH = "models/customer_segmentation_kmeans.pkl"
SCALER_PATH = "models/customer_segmentation_scaler.pkl"
PCA_PATH = "models/customer_segmentation_pca.pkl"

kmeans = joblib.load(KMEANS_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

print("=" * 50)
print("SCALER")
print("=" * 50)
print("Type:", type(scaler).__name__)
print("Input features:", scaler.n_features_in_)
print("Feature names:", getattr(scaler, "feature_names_in_", "Not saved"))

print("" + "=" * 50)
print("PCA")
print("=" * 50)
print("Input features:", pca.n_features_in_)
print("Output components:", pca.n_components_)

print("" + "=" * 50)
print("K-MEANS")
print("=" * 50)
print("Input features:", kmeans.n_features_in_)
print("Number of clusters:", kmeans.n_clusters)

print("" + "=" * 50)
print("PIPELINE CONCLUSION")
print("=" * 50)

if kmeans.n_features_in_ == pca.n_components_:
    print("PCA WAS USED before K-Means.")
else:
    print("PCA WAS NOT USED before K-Means.")
    print("Production pipeline: RobustScaler -> K-Means")


print("Scaler feature names:")
print(scaler.feature_names_in_)
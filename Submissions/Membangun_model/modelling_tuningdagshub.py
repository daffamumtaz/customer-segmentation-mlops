import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import dagshub

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from mlflow.models.signature import infer_signature

# for assignment convenience. Will revoke after this program ends
MY_TOKEN = "a379ada3f8ea88155a3c4e1f07c4399f62e8cfe3"

dagshub.auth.add_app_token(MY_TOKEN)

dagshub.init(
    repo_owner="Neuraly4",
    repo_name="SMSML-Daffa-Naufal-Mumtaz-Heryadi",
    mlflow=True
)

mlflow.set_experiment("Mall_Customer_Clustering_Advance")

mlflow.sklearn.autolog(disable=True)

df = pd.read_csv("Mall_Customers_preprocessed.csv")
X = df.select_dtypes(include="number")

candidate_k = [2, 3, 4, 5, 6, 7, 8]

best_score = -1
best_model = None
best_k = None

inertias = []
sil_scores = []

for k in candidate_k:
    with mlflow.start_run(run_name=f"k={k}"):

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels)
        inertia = model.inertia_

        inertias.append(inertia)
        sil_scores.append(sil)

        mlflow.log_param("n_clusters", k)
        mlflow.log_param("init", "k-means++")
        mlflow.log_param("n_init", 10)
        mlflow.log_param("max_iter", 300)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("inertia", inertia)

        signature = infer_signature(X, labels)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X.iloc[:5]
        )

        if sil > best_score:
            best_score = sil
            best_model = model
            best_k = k
            
# Elbow Plot (Arte 1)

plt.figure()
plt.plot(candidate_k, inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.savefig("elbow_plot.png")
mlflow.log_artifact("elbow_plot.png")
plt.close()

# PCA (Arte 2)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
labels = best_model.predict(X)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title(f"Cluster Visualization (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("cluster_visualization.png")
mlflow.log_artifact("cluster_visualization.png")
plt.close()

# Best Model Summary (Arte Basic)

with mlflow.start_run(run_name="Best_Model_Summary"):
    mlflow.log_param("best_k", best_k)
    mlflow.log_metric("best_silhouette_score", best_score)

    final_labels = best_model.predict(X)
    signature = infer_signature(X, final_labels)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        signature=signature,
        input_example=X.iloc[:5]
    )

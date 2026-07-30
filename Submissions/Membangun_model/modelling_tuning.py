import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlflow.models.signature import infer_signature

mlflow.set_experiment("Mall_Customer_Clustering")
mlflow.sklearn.autolog(disable=True) 

df = pd.read_csv("Mall_Customers_preprocessed.csv")
X = df.select_dtypes(include="number")

cluster_range = [2, 3, 4, 5, 6, 7, 8]
best_score = -1
best_model = None
best_k = None

for k in cluster_range:
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

        mlflow.log_param("n_clusters", k)
        mlflow.log_param("init", "k-means++")
        mlflow.log_param("n_init", 10)
        mlflow.log_param("max_iter", 300)
        mlflow.log_metric("silhouette_score", sil)

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

print(f"Best K: {best_k}, Best Silhouette: {best_score:.4f}")

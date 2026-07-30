import pandas as pd
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn

df = pd.read_csv("Mall_Customers_preprocessed.csv")

X = df.values

mlflow.set_experiment("Mall_Customer_Clustering")

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X)

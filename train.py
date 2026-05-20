import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Chargement
df = pd.read_csv("./datasets/fruits.csv", header=None, names=["x", "y"])
X = df.values

# 2. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Clustering (k=3 déterminé par méthode du coude)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

# 4. Sauvegarde
joblib.dump(model, "./models/model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("Modèle et scaler sauvegardés : model.pkl, scaler.pkl")
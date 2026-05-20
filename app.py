from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)

# Chargement des artefacts (depuis le répertoire courant)
model = joblib.load("./models/model.pkl")
scaler = joblib.load("./models/scaler.pkl")

app = FastAPI(title="Fruit Clustering API", description="API non supervisée - KMeans", version="1.0")

class Features(BaseModel):
    x: float
    y: float

@app.get("/")
def read_root():
    return {"message": "API de clustering - utilisez POST /predict"}

@app.post("/predict")
def predict(features: Features):
    try:
        logging.info(f"Requête reçue : {features}")
        data = np.array([[features.x, features.y]])
        data_scaled = scaler.transform(data)
        cluster = int(model.predict(data_scaled)[0])
        # Option : renvoyer aussi les coordonnées des centres
        return {"cluster": cluster, "message": f"Point assigné au cluster {cluster}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
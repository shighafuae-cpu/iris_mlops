from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import joblib
from pathlib import Path
import json

app = FastAPI(title="Iris Classifier API", description="API for classifying Iris species using a trained Logistic Regression model.", version="1.0.0")


MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)

def load_target_names():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found at {MODEL_PATH}. Please train the model first.")
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)
    
    return data.get("target_names", [])

#----------------------------------
# Health Check
#----------------------------------
@app.get("/")
def home():
    return {"status": "API is running"}

# -----------------------------
# Request Model
# -----------------------------
class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictRequest):
    # Placeholder for prediction logic
    try:
        features = request.features
        if not isinstance(features, list) or len(features) != 4:
            raise HTTPException(status_code=400, detail="Input must be a list of 4 numerical features.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for input features.")
    
    X = np.array(features).reshape(1, -1)
    model = load_model()
    pred = model.predict(X)
    target_names = load_target_names()

    return {"prediction": target_names[pred[0]]}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
from pathlib import Path
import os
import time

import mlflow
import mlflow.sklearn as mlflowSklearn

from fastapi.middleware.cors import CORSMiddleware

# MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_VERSION = "v2"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
MODEL_NAME = "text-classifier"
MODEL_STAGE = "Production"

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
script_directory = script_directory / f"model/{MODEL_VERSION}"

print(f"Current file's directory: {script_directory}")

artifact_path = script_directory / "artifact.pkl"
# model_file_path = script_directory / "model.pkl"
vectorizer_file_path = script_directory / "vectorizer.pkl"

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

print(f"artifact_path: {artifact_path}")
# print(f"model_path: {model_file_path}")
print(f"model_uri: {model_uri}")
print(f"vectorizer_file_path: {vectorizer_file_path}")

# Load model and vectorizer at startup
artifact = joblib.load(artifact_path)
# model = joblib.load(model_file_path)
model = mlflowSklearn.load_model(model_uri)
vectorizer = joblib.load(vectorizer_file_path)

# Init model artifact information
model_name = artifact["model"]
vectorizer_name = artifact["vectorizer"]
labels = artifact["labels"]
sklearn_version = artifact["sklearn_version"]
model_created_at = artifact["created_at"]

app = FastAPI(title="AI Text Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS
    allow_headers=["*"],
)

# Define request schema - Input for AI model to classify
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text to classify",
        examples=["I love this product"]
    )

class ModelInfoResponse(BaseModel):
    model_type: str
    labels:List[str]
    vectorizer: str
    version: str
    sklearn_version: str
    created_at: str

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    sentiments_str = ["Negative", "Neutral", "Positive"]
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        # Vectorize input text
        text_vector = vectorizer.transform([request.text])

        # Predict label
        prediction = sentiments_str[model.predict(text_vector)[0]]

        # Get confidence
        probability = model.predict_proba(text_vector)[0]
        confidence = float(probability.max())

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "latency_ms": latency_ms,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ïnference failed: {str(e)}"
        )

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse) # response_model tells FastAPI the data shape
def get_model_info():
    return {
        "model_type": type(model).__name__,
        "labels": labels,
        "vectorizer": type(vectorizer).__name__,
        "version": artifact["version"],
        "sklearn_version": sklearn_version,
        "created_at": model_created_at
    }

# Backend health API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }

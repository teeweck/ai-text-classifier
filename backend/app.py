from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
from pathlib import Path
import os
import time

import mlflow
import mlflow.sklearn as mlflowSklearn
from mlflow.tracking import MlflowClient

from fastapi.middleware.cors import CORSMiddleware

# MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_VERSION = "v1"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
MODEL_NAME = "text-classifier"
MODEL_STAGE = "Production"

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
script_directory = script_directory / f"model/{MODEL_VERSION}"

print(f"Current file's directory: {script_directory}")

# model_file_path = script_directory / "model.pkl"
vectorizer_file_path = script_directory / "vectorizer.pkl"

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# print(f"model_path: {model_file_path}")
print(f"model_uri: {model_uri}")
print(f"vectorizer_file_path: {vectorizer_file_path}")

# Initialize the MlflowClient
client = MlflowClient()

# The stage you are interested in (e.g., "Production")
stage = "Production"

# model_versions is a list
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for mv in model_versions:
    if mv.current_stage == "Production":
        MODEL_VERSION = mv.version
        run_id = mv.run_id
        if run_id == None:
            raise HTTPException(status_code=400, detail=f"run_id not found")
        run = client.get_run(run_id)
        # print(f"Run ID: {run.info.run_id}")
        # print(f"Parameters: {run.data.params}")
        # print(f"Metrics: {run.data.metrics}")
        break

# Load model and vectorizer at startup
# model = joblib.load(model_file_path)
model = mlflowSklearn.load_model(model_uri)
vectorizer = joblib.load(vectorizer_file_path)

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
        # text_vector = vectorizer.transform([request.text])
        text_vector = [request.text]

        # Predict label
        prediction_ind = model.predict(text_vector)[0]
        prediction = sentiments_str[prediction_ind]

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
    model_artifacts = run.data.params
    return {
        "model_type": model_artifacts["model_type"],
        "labels": model_artifacts["labels"].split(", "),
        "vectorizer": model_artifacts["vectorizer"],
        "version": MODEL_VERSION,
        "sklearn_version": model_artifacts["sklearn_version"],
        "created_at": model_artifacts["created_at"]
    }

# Backend health API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }

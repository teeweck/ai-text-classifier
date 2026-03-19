from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from typing import List
import time
import sys

import mlflow
import mlflow.sklearn as mlflowSklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

MODEL_VERSION = "v1"
SERVER_URL = "http://127.0.0.1:5000/"
MODEL_NAME = "text-classifier"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(SERVER_URL)
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
print(f"model_uri: {model_uri}")

# Initialize the MlflowClient
client = MlflowClient()

try:
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    found = False
    for mv in model_versions:
        if mv.current_stage == MODEL_STAGE:
            MODEL_VERSION = mv.version
            run_id = mv.run_id
            if run_id is None:
                print("run_id not found for the model in Production stage.")
                sys.exit(1)
            run = client.get_run(run_id)
            found = True
            break
    if not found:
        print(f"No model named '{MODEL_NAME}' in Production stage found in MLflow registry.")
        sys.exit(1)
except RestException as e:
    print(f"MLflow error: {e}")
    print(f"Registered Model with name={MODEL_NAME} not found. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)

# Load model pipeline at startup
model = mlflowSklearn.load_model(model_uri)

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

        model_input = [request.text]

        # Predict label
        prediction_ind = model.predict(model_input)[0]
        prediction = sentiments_str[prediction_ind]

        # Get confidence
        probability = model.predict_proba(model_input)[0]
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
    }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import os
import time

from fastapi.middleware.cors import CORSMiddleware

model_file_path = os.path.dirname(__file__) + "/model/model.pkl"
vectorizer_file_path = os.path.dirname(__file__) + "/model/vectorizer.pkl"

# Load model and vectorizer at startup
model = joblib.load(model_file_path)
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

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.perf_counter()

    try:
        # Vectorize input text
        text_vector = vectorizer.transform([request.text])

        # Predict label
        prediction = model.predict(text_vector)[0]

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
        "labels": list(model.classes_),
        "vectorizer": "TF-IDF",
        "version": "1.0.0"
    }

# Backend health API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }

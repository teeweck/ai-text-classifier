from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import joblib

# Load model and vectorizer at startup
model = joblib.load("../model/model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

app = FastAPI(title="AI Text Classifier")

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
    # Vectorize input text
    text_vector = vectorizer.transform([request.text])

    # Predict label
    prediction = model.predict(text_vector)[0]

    # Get confidence
    confidence = max(model.predict_proba(text_vector)[0])

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse) # response_model tells FastAPI the data shape
def get_model_info():
    return {
        "model_type": type(model).__name__,
        "labels": list(model.classes_),
        "vectorizer": "TF-IDF",
        "version": "1.0.0"
    }

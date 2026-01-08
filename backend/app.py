from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Load model and vectorizer at startup
model = joblib.load("../model/model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

app = FastAPI(title="AI Text Classifier")

# Define request schema
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text to classify",
        examples=["I love this product"]
    )

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

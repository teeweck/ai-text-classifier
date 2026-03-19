import pandas as pd
import mlflow
import mlflow.sklearn as mlflowSklearn
import sklearn
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# Get program local directory
current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent

# Load dataset
dataset_location = "data/raw/sentiment_data.csv"
data = pd.read_csv(dataset_location)

# Sentiment mapping
# 0 — Negative
# 1 — Neutral
# 2 — Positive
sentiment_counts = data['Sentiment'].value_counts()
# print(f"Sentiment counts:\n{sentiment_counts}")

texts = data["Comment"]
labels = data["Sentiment"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)

# Text vectorisation
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

# Set experiment name
mlflow.set_experiment("text-classifier")
saveModelToRegistry = False

# Model parameters
max_model_iter = 1000
MODEL_NAME = "text-classifier"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=max_model_iter))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f"Model accuracy: {accuracy}")

with mlflow.start_run():
    req_path = current_directory / f"requirements.txt"
    if (saveModelToRegistry == True):
        mlflowSklearn.log_model(
            pipeline, 
            name="sentiment analysis model", 
            serialization_format='skops',
            pip_requirements=[f"-r {req_path}"],
            registered_model_name=MODEL_NAME
        )
    else:
        print("Trained model not saved into registry")

    vectorizer_type = type(pipeline.named_steps["vectorizer"]).__name__
    classifier_type = type(pipeline.named_steps["classifier"]).__name__

    labels_str = [label for label in sentiment_counts.keys()][::-1]
    print(str(labels_str)[1:-1])

    # Log parameters
    mlflow.log_param("model_type", classifier_type)
    mlflow.log_param("max_iter", max_model_iter)
    mlflow.log_param("vectorizer", vectorizer_type)
    mlflow.log_param("labels", str(labels_str)[1:-1])
    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("created_at", datetime.now().isoformat())

    # Log metrics
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("f1_score", float(f1))

    # Log training dataset
    mlflow.log_artifact(dataset_location)

print(f"Accuracy: {accuracy:.2f}")
print(f"f1: {f1:.2f}")
print(classification_report(y_test, predictions))
print("Training complete")  

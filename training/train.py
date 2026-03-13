import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import sklearn
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Load dataset
data = pd.read_csv("data/raw/sentiment_data.csv")

sentiment_counts = data['Sentiment'].value_counts()
print(f"Sentiment counts:\n{sentiment_counts}")

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

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Set experiment name
mlflow.set_experiment("text-classifier")

# Model parameters
max_model_iter = 1000

with mlflow.start_run():
    # Model training
    model = LogisticRegression(max_iter=max_model_iter)
    model.fit(X_train_vec, y_train)

    # Model evaluation
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", max_model_iter)
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("sklearn_version", sklearn.__version__)

    # Log metrics
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("f1_score", float(f1))

    # Log model artifact
    mlflow.sklearn.log_model(model, name="sentiment analysis model")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"f1: {f1:.2f}")
    print(classification_report(y_test, predictions))

# Save Model artifacts
MODEL_VERSION = "v2"

labels_str = [str(label) for label in model.classes_.tolist()]

artifact = {
    "model": model,
    "vectorizer": vectorizer,
    "labels": labels_str,
    "sklearn_version": sklearn.__version__,
    "created_at": datetime.now().isoformat(),
    "version": MODEL_VERSION,
}

current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent.parent
new_dir = current_directory / f"backend/model/{MODEL_VERSION}"

try:
    new_dir.mkdir()
    print(f"Folder '{new_dir}' created.")
except:
    print(f"Folder '{new_dir}' already exists.")

joblib.dump(artifact, f"{new_dir}/artifact.pkl")
joblib.dump(model, f"{new_dir}/model.pkl")
joblib.dump(vectorizer, f"{new_dir}/vectorizer.pkl")
print("Model,vectorizer and artifacts saved successfully.")

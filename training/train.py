import pandas as pd
import joblib
import sklearn
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/raw/sentiment_data.csv")

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

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Model evaluation
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

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

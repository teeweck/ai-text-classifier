import pandas as pd
import mlflow
import mlflow.sklearn as mlflowSklearn
from mlflow.tracking import MlflowClient
import sklearn
from datetime import datetime
from pathlib import Path
import json

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# Get program local directory
current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent

def load_config(file_path):
    # Check if file exists to avoid errors
    if not Path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

config_data = load_config(current_directory / f"config.json")
# Check if config was loaded successfully
if config_data is None:
    print("Error: Configuration file could not be loaded. Exiting.")
    exit(1)

def train_function():
    print("train.py function")
    return "train.py function"

# Model parameters
MODEL_NAME = config_data["model_config"]["model_name"]
mlflow.set_experiment(MODEL_NAME)
saveModelToRegistry = False
max_model_iter = config_data["model_config"]["max_model_iter"]

mlflow_ip = config_data["model_config"]["mlflow_ip"]
mlflow_port = config_data["model_config"]["mlflow_port"]
mlflow.set_tracking_uri("http://" + f"{mlflow_ip}" + ":" + f"{mlflow_port}")

client = MlflowClient()
production_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

production_accuracy = None

for mv in production_versions:
    if mv.current_stage == "Production":
        MODEL_VERSION = mv.version
        run_id = mv.run_id
        if run_id is None:
            print("run_id not found for the model in Production stage.")
            exit(1)
        run = client.get_run(run_id)
        production_accuracy = run.data.metrics.get("accuracy")
        break

# Load dataset
dataset_location = config_data["app_settings"]["dataset_path"]
data = pd.read_csv(dataset_location)

sentiment_counts = data['Sentiment'].value_counts()

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

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=max_model_iter))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

should_promote = False

if production_accuracy is None:
    print("No production model found. Promoting new model.")
    should_promote = True
elif accuracy > production_accuracy + config_data["model_config"]["accuracy_improvement_buf"]:
    print("New model is better. Promoting.")
    should_promote = True
else:
    print("New model is worse. Keeping current production model.")

# Define the input schema (the features)
input_schema = Schema([
    ColSpec("string", "text", required=True) # Optional field
])

# Define the output schema (the prediction)
output_schema = Schema([
    ColSpec("integer", "prediction"),
    ColSpec("double", "confidence"),
    ColSpec("double", "latency_ms")
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run():
    req_path = current_directory / f"requirements.txt"
    if (config_data["model_config"]["save_model"] == "true"):
        mlflowSklearn.log_model(
            pipeline, 
            name="sentiment analysis model", 
            serialization_format='skops',
            pip_requirements=[f"-r {req_path}"],
            registered_model_name=MODEL_NAME,
            signature=signature,
        )
        latest_version = client.search_model_versions(f"name='{MODEL_NAME}'")[0]
        print(latest_version)
        if should_promote:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version.version,
                stage="Production"
            )
    else:
        print("Trained model not saved into registry")

    vectorizer_type = type(pipeline.named_steps["vectorizer"]).__name__
    classifier_type = type(pipeline.named_steps["classifier"]).__name__

    labels_str = [label for label in sentiment_counts.keys()][::-1]

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

if (config_data["app_settings"]["print_debug_msg"] == "true"):
    print(f"Accuracy: {accuracy:.2f}")
    print(f"f1: {f1:.2f}")
    print(classification_report(y_test, predictions))
print("Training complete")  

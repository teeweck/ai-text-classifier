import pandas as pd
import mlflow
import mlflow.sklearn as mlflowSklearn
from mlflow.tracking import MlflowClient
from mlflow.store.entities.paged_list import PagedList
from mlflow.entities.model_registry.model_version import ModelVersion

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

client = MlflowClient()
config_data = None

def load_config(file_path):
    global config_data
    # Check if file exists to avoid errors
    if not Path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    with open(file_path, 'r') as file:
        config_data = json.load(file)

    return config_data

def get_stage_latest_run(experiment_paged_list: PagedList[ModelVersion], stage: str):
    run = None
    for mv in experiment_paged_list:
        if mv.current_stage == stage:
            run_id = mv.run_id
            if run_id is None:
                print("run_id not found for the model in Production stage.")
                exit(1)
            run = client.get_run(run_id)
            break

    return run

def promote_model(production_accuracy, new_accuracy) -> bool:
    promote = False
    if config_data is None:
        print("promote_model error: config_data not found")
        exit(1)

    if production_accuracy is None:
        print("No production model found. Promoting new model.")
        promote = True
    elif new_accuracy > production_accuracy + config_data["model_config"]["accuracy_improvement_buf"]:
        print("New model is better. Promoting.")
        promote = True
    else:
        print("New model is worse. Keeping current production model.")
    
    return promote

def model_signature() -> ModelSignature:
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

    return signature

def train_model():
    # Check if config was loaded successfully
    if config_data is None:
        print("Error: Configuration file could not be loaded. Exiting.")
        exit(1)

    # Model parameters
    MODEL_NAME = config_data["model_config"]["model_name"]
    MODEL_STAGE = config_data["model_config"]["model_stage"]
    
    max_model_iter = config_data["model_config"]["max_model_iter"]
    mlflow_ip = config_data["model_config"]["mlflow_ip"]
    mlflow_port = config_data["model_config"]["mlflow_port"]

    mlflow.set_experiment(MODEL_NAME)
    mlflow.set_tracking_uri("http://" + f"{mlflow_ip}" + ":" + f"{mlflow_port}")

    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    run = get_stage_latest_run(model_versions, MODEL_STAGE)
    if run is None:
        print("Error: run not found")
        exit(1)

    production_accuracy = run.data.metrics.get("accuracy")

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
        test_size=config_data["model_config"]["test_size"],
        random_state=config_data["model_config"]["random_state"]
    )

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000)),
        ("classifier", LogisticRegression(max_iter=max_model_iter))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    should_promote = promote_model(production_accuracy, accuracy)
    signature = model_signature()

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

            if should_promote:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=latest_version.version,
                    stage=MODEL_STAGE
                )
            print("Trained model saved into registry")
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
    return 0

if __name__ == "__main__":
    load_config(current_directory / f"config.json")

    train_model()
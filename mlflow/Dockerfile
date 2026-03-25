FROM python:3.13-slim

WORKDIR /mlflow

# Install MLflow
RUN pip install mlflow==3.10.0

# Create directories for the database and artifacts
RUN mkdir -p /mlflow/db /mlflow/artifacts

# Expose the default MLflow port
EXPOSE 5000

# Run MLflow server
# Using SQLite for metadata and local directory for artifacts
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
    "--default-artifact-root", "/mlflow/artifacts"]
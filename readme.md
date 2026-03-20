# AI Text Classifier

A small project to train and serve a text classification model. The purpose of the project is to understand how to deploy a simple AI model.

## Project structure

```text
ai-text-classifier/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│
├── model/
│   └── model.pkl
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

## Run the backend

From the `backend/` directory, run:

```bash
uvicorn app:app --reload --port 8000
```

To restart the server:

```bash
uvicorn app:app --reload
```

Open the API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## What you get

- Live API
- Auto-generated Swagger UI
- Interactive testing

## Mental model

Endpoint = Function\
Decorator = Routing + HTTP semantics\
Pydantic Model = Contract\
FastAPI = Glue\
Swagger UI = Visualization

## Locally run backend

Command to run backend (app.py) locally:
`uvicorn backend.app:app --reload --port 8000`

Alternative command to run backend (app.py) locally:
`fastapi run .\backend\app.py`

## Docker for backend

Building the Docker Image:

- Reads `backend/Dockerfile`
- Builds an immutable image
- Tags it as `ai-backend`

From the `root` directory, run:

```bash
docker compose build backend
```

Run the container

- Runs the backend container in detached mode (in the background)
- Only runs the backend service

```bash
docker compose up -d backend
```

## Docker for frontend

Building the Docker Image (frontend):

From the `root` directory, run:

```bash
docker compose build frontend
```

Run the container

```bash
docker compose up -d frontend
```

Open browser:

```bash
http://localhost:3000
```

## Key docker commands

- To check status of docker service: `docker compose ps`
- To down a docker service: `docker compose down <service name>`

## Docker compose

Running docker compose (backend + frontend):

From the `root` directory, run:

```bash
docker compose up --build
```

Docker will:

1. Build backend image
2. Build frontend image
3. Created shared network
4. Start both containers

## MLFlow

MLflow is an open-source platform that helps streamline and organize the process of developing, tracking, and deploying machine learning models.

MLFlow server needs to be run to provide backend with access to the ML model.

Command to run MLFlow server:\
`mlflow server --host 127.0.0.1 --port 5000`\
-or- \
`mlflow ui`

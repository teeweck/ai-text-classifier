# AI Text Classifier

A small project to train and serve a text classification model.

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

## Docker for backend

Building the Docker Image:

- Reads `backend/Dockerfile`
- Builds an immutable image
- Tags it as `ai-backend`

From the `root` directory, run:

```bash
docker build -t ai-backend -f backend/Dockerfile .
```

Run the container

```bash
docker run -p 8000:8000 ai-backend

mapping:
host:8000 → container:8000
```

## Docker for frontend

Building the Docker Image (frontend):

From the `root` directory, run:

```bash
docker build -t ai-frontend -f frontend/Dockerfile .
```

Run the container

```bash
docker run -p 3000:3000 ai-frontend
```

Open browser:

```bash
http://localhost:3000
```

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

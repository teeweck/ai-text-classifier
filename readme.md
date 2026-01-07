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

Open the API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## What you get

- Live API
- Auto-generated Swagger UI
- Interactive testing
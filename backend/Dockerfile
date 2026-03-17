# 1️⃣ Base image
FROM python:3.13-slim

ENV MODEL_VERSION=v2

# 2️⃣ Set working directory inside container
WORKDIR /app

# Install curl and other system dependencies
RUN apt-get update && apt-get install -y curl

# 3️⃣ Copy dependency list first (for caching)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy backend code
COPY app.py .

# 6️⃣ Copy trained model artifacts
# COPY model ./model

# 7️⃣ Expose port
EXPOSE 8000

# 8️⃣ Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

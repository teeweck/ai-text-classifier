from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

class TestApp:
    def test_predict_valid(self):
        response = client.post("/predict", json={"text": "This is great!"})
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "confidence" in response.json()

    def test_predict_invalid(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_model_info(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        assert "model_type" in response.json()
        assert "labels" in response.json()

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] == True
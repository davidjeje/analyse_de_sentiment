import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.api import app

# comment
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}

def test_get_tweets():
    response = client.get("/tweets?sample_frac=0.01")
    assert response.status_code == 200
    data = response.json()
    assert "tweets" in data
    assert isinstance(data["tweets"], list)
    assert len(data["tweets"]) > 0

def test_get_tweets_file_missing():
    with patch("os.path.exists", return_value=False):
        response = client.get("/tweets?sample_frac=0.01")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

def test_predict_sentiment_positive():
    payload = {"text": "I love this product"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] in ["positive", "neutral", "negative", "unknown"]
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_sentiment_empty_text():
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    # Validation Pydantic min_length=1 doit renvoyer 422 Unprocessable Entity
    assert response.status_code == 422
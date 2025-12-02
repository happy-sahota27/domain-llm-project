"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Domain LLM API" in response.json()["message"]


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


def test_health_response_structure():
    """Test health response structure."""
    response = client.get("/api/v1/health")
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_model_info_without_model():
    """Test model info endpoint when no model is loaded."""
    response = client.get("/api/v1/model/info")
    
    # Should return 503 when model not loaded
    assert response.status_code == 503


def test_generate_without_model():
    """Test generate endpoint when no model is loaded."""
    response = client.post("/api/v1/generate", json={
        "prompt": "Test prompt",
        "max_tokens": 50
    })
    
    # Should return 503 when model not loaded
    assert response.status_code == 503


def test_generation_request_validation():
    """Test request validation."""
    # Valid request structure (will fail without model, but validates schema)
    response = client.post("/api/v1/generate", json={
        "prompt": "Test",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    })
    
    # 503 because no model, but request was valid
    assert response.status_code == 503


def test_invalid_temperature():
    """Test invalid temperature parameter."""
    response = client.post("/api/v1/generate", json={
        "prompt": "Test",
        "temperature": 3.0  # Invalid: > 2.0
    })
    
    # Should return 422 for validation error
    assert response.status_code == 422


def test_batch_generation_without_model():
    """Test batch generation endpoint."""
    response = client.post("/api/v1/generate/batch", json={
        "prompts": ["Test 1", "Test 2"],
        "max_tokens": 50
    })
    
    assert response.status_code == 503


def test_rerank_without_reranker():
    """Test rerank endpoint when reranker not loaded."""
    response = client.post("/api/v1/rerank", json={
        "query": "test query",
        "documents": ["doc1", "doc2"],
        "top_k": 1
    })
    
    assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

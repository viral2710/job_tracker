import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    model_ids = [m["model_id"] for m in data]
    assert "tfidf" in model_ids
    assert "keyword" in model_ids
    assert "semantic" in model_ids

def test_get_jobs_endpoint():
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) > 0
    assert "title" in jobs[0]

def test_match_endpoint():
    payload = {
        "candidate": {
            "candidate_id": "test_candidate",
            "full_name": "Test Engineer",
            "skills": ["Python", "FastAPI"],
            "experience_summary": "Python and API developer",
            "preferred_roles": ["Backend Developer"],
            "preferred_locations": ["Remote"]
        },
        "model_name": "tfidf",
        "top_k": 3
    }

    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["model_used"] == "tfidf"
    assert len(res["matches"]) <= 3
    assert res["execution_time_ms"] >= 0

def test_match_invalid_model():
    payload = {
        "candidate": {
            "candidate_id": "test_candidate",
            "full_name": "Test",
            "skills": ["Python"],
            "experience_summary": "Python developer",
            "preferred_roles": [],
            "preferred_locations": []
        },
        "model_name": "non_existent_model_123"
    }

    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 400

def test_scrape_endpoint_sample():
    response = client.post("/api/v1/scrape", json={"url": "sample", "replace_existing": False})
    assert response.status_code == 200
    data = response.json()
    assert data["jobs_fetched"] > 0

def test_metrics_endpoints():
    # Fetch summary and history
    sum_resp = client.get("/api/v1/metrics/summary")
    assert sum_resp.status_code == 200
    
    hist_resp = client.get("/api/v1/metrics/history")
    assert hist_resp.status_code == 200

def test_ui_index_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Career Site Job Tracker" in response.text


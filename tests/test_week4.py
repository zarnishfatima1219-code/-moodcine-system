"""
MoodCine — Unit Tests: Week 4 (FER + Fusion + FastAPI)
=======================================================
Run: python -m pytest tests/test_week4.py -v
"""
import pytest
import sys
sys.path.append('.')

# ── FUSION LAYER TESTS ─────────────────────────────────────────────────
from backend.ml.fusion.combine import fuse_emotions

def test_fusion_returns_dominant():
    """Fusion returns a dominant emotion"""
    nlp = {"fear":0.9,"joy":0.05,"neutral":0.05,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    fer = {"neutral":0.6,"joy":0.3,"fear":0.1,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    result = fuse_emotions(nlp, fer)
    assert "dominant_emotion" in result
    assert "confidence" in result
    assert "fused_scores" in result

def test_fusion_weights_correct():
    """NLP 60% + FER 40% weights applied correctly"""
    nlp = {"fear":1.0,"joy":0.0,"neutral":0.0,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    fer = {"joy":1.0,"fear":0.0,"neutral":0.0,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    result = fuse_emotions(nlp, fer, nlp_weight=0.6, fer_weight=0.4)
    assert result["fused_scores"]["fear"] > result["fused_scores"]["joy"]

def test_fusion_scores_sum_to_one():
    """Fused scores sum to 1.0"""
    nlp = {"fear":0.9,"joy":0.05,"neutral":0.05,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    fer = {"neutral":0.7,"joy":0.2,"fear":0.1,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    result = fuse_emotions(nlp, fer)
    total = sum(result["fused_scores"].values())
    assert abs(total - 1.0) < 0.01

def test_fusion_dominant_is_highest():
    """Dominant emotion has highest score"""
    nlp = {"joy":0.8,"neutral":0.1,"fear":0.1,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    fer = {"joy":0.9,"neutral":0.1,"fear":0.0,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    result = fuse_emotions(nlp, fer)
    dominant = result["dominant_emotion"]
    assert result["fused_scores"][dominant] == max(result["fused_scores"].values())

def test_fusion_all_emotions_present():
    """All 7 emotions in fused output"""
    nlp = {"fear":0.9,"joy":0.05,"neutral":0.05,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    fer = {"neutral":0.6,"joy":0.3,"fear":0.1,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    result = fuse_emotions(nlp, fer)
    expected = {"anger","disgust","fear","joy","neutral","sadness","surprise"}
    assert set(result["fused_scores"].keys()) == expected

def test_fusion_custom_weights():
    """Custom weights work correctly"""
    nlp = {"joy":1.0,"fear":0.0,"neutral":0.0,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    fer = {"joy":1.0,"fear":0.0,"neutral":0.0,"sadness":0.0,"anger":0.0,"disgust":0.0,"surprise":0.0}
    result = fuse_emotions(nlp, fer, nlp_weight=0.7, fer_weight=0.3)
    assert result["dominant_emotion"] == "joy"

# ── FASTAPI TESTS ──────────────────────────────────────────────────────
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_health_models_loaded():
    """Health check confirms models loaded"""
    response = client.get("/health")
    data = response.json()
    assert data["nlp_model"] == "loaded"
    assert data["svd_model"] == "loaded"
    assert data["movies_count"] == 1682

def test_root_endpoint():
    """Root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "MoodCine API"

def test_nlp_emotion_endpoint():
    """NLP emotion endpoint works"""
    response = client.post("/nlp-emotion",
        json={"mood_text": "I feel really happy today"})
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] == "joy"
    assert data["confidence"] > 50

def test_nlp_emotion_fear():
    """NLP detects fear correctly"""
    response = client.post("/nlp-emotion",
        json={"mood_text": "I feel scared and anxious"})
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] == "fear"

def test_recommend_endpoint():
    """Recommend endpoint returns 5 movies"""
    response = client.post("/recommend",
        json={"mood_text": "I feel happy", "mode": "nlp"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5

def test_recommend_has_poster():
    """Recommendations include poster URLs"""
    response = client.post("/recommend",
        json={"mood_text": "I feel sad", "mode": "nlp"})
    data = response.json()
    movies_with_posters = [m for m in data["recommendations"] if m["poster_url"]]
    assert len(movies_with_posters) > 0

def test_recommend_baseline_mode():
    """Baseline mode works"""
    response = client.post("/recommend",
        json={"mood_text": "test", "mode": "baseline"})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 5

def test_movies_endpoint():
    """Movies list endpoint works"""
    response = client.get("/movies?limit=5")
    assert response.status_code == 200
    assert len(response.json()["movies"]) == 5

def test_single_movie_endpoint():
    """Single movie endpoint works"""
    response = client.get("/movies/1")
    assert response.status_code == 200
    data = response.json()
    assert data["movie_id"] == 1
    assert "Toy Story" in data["title"]

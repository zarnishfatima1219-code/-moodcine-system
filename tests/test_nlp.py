"""
MoodCine — Unit Tests: NLP Emotion Module
==========================================
Run: python -m pytest tests/test_nlp.py -v
"""
import sys
sys.path.append('.')
from transformers import pipeline
import pytest

@pytest.fixture(scope="module")
def emotion_classifier():
    # top_k=None returns ALL scores (works with newer transformers versions)
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

def get_scores(classifier, text):
    """Helper: returns dict of {emotion: score}"""
    result = classifier(text)[0]  # [0] gives list of dicts
    return {r["label"]: r["score"] for r in result}

def test_model_loads(emotion_classifier):
    """Test NLP model loads successfully"""
    assert emotion_classifier is not None

def test_joy_detected(emotion_classifier):
    """Test joy emotion correctly detected"""
    scores = get_scores(emotion_classifier, "I am so happy and excited today!")
    assert scores["joy"] == max(scores.values()), "Joy should be dominant"

def test_fear_detected(emotion_classifier):
    """Test fear emotion correctly detected"""
    scores = get_scores(emotion_classifier, "I feel really anxious and scared")
    assert scores["fear"] == max(scores.values()), "Fear should be dominant"

def test_sadness_detected(emotion_classifier):
    """Test sadness emotion correctly detected"""
    scores = get_scores(emotion_classifier, "I feel so sad and lonely today")
    assert scores["sadness"] == max(scores.values()), "Sadness should be dominant"

def test_anger_detected(emotion_classifier):
    """Test anger emotion correctly detected"""
    scores = get_scores(emotion_classifier, "I am furious and cannot calm down")
    assert scores["anger"] == max(scores.values()), "Anger should be dominant"

def test_output_has_7_emotions(emotion_classifier):
    """Test model returns all 7 emotion scores"""
    result = emotion_classifier("I feel okay today")[0]
    assert len(result) == 7, f"Expected 7 emotions, got {len(result)}"

def test_scores_sum_to_one(emotion_classifier):
    """Test emotion scores sum to approximately 1.0"""
    result = emotion_classifier("feeling good")[0]
    total = sum(r["score"] for r in result)
    assert abs(total - 1.0) < 0.01, "Scores should sum to ~1.0"

def test_all_7_emotions_present(emotion_classifier):
    """Test all 7 expected emotion labels are returned"""
    expected = {"anger","disgust","fear","joy","neutral","sadness","surprise"}
    result = emotion_classifier("test")[0]
    labels = {r["label"] for r in result}
    assert labels == expected, f"Missing emotions: {expected - labels}"

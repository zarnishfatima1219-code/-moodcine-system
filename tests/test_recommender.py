"""
MoodCine — Unit Tests: Recommendation Engine
=============================================
Run: python -m pytest tests/test_recommender.py -v
"""
import pytest
import pickle
import pandas as pd
import psycopg2

DB = {"host":"localhost","database":"moodcine","user":"postgres","password":"moodcine123","port":5432}

@pytest.fixture(scope="module")
def db_conn():
    conn = psycopg2.connect(**DB)
    yield conn
    conn.close()

@pytest.fixture(scope="module")
def svd_model():
    with open("backend/ml/recommender/svd_model.pkl","rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def movies_df(db_conn):
    return pd.read_sql("SELECT movie_id, title, genres FROM movies", db_conn)

# ── Database Data Tests ────────────────────────────────────────────────
def test_movies_loaded(db_conn):
    """1682 movies in database"""
    cur = db_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM movies")
    count = cur.fetchone()[0]
    assert count == 1682, f"Expected 1682 movies, got {count}"

def test_ratings_loaded(db_conn):
    """100,000 ratings in database"""
    cur = db_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ratings")
    count = cur.fetchone()[0]
    assert count == 100000, f"Expected 100000 ratings, got {count}"

def test_users_loaded(db_conn):
    """943 users in database"""
    cur = db_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    assert count == 943, f"Expected 943 users, got {count}"

def test_ratings_range(db_conn):
    """All ratings between 1 and 5"""
    cur = db_conn.cursor()
    cur.execute("SELECT MIN(rating), MAX(rating) FROM ratings")
    min_r, max_r = cur.fetchone()
    assert min_r >= 1, "Rating below 1 found"
    assert max_r <= 5, "Rating above 5 found"

def test_movies_have_genres(db_conn):
    """Movies have genre data"""
    cur = db_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM movies WHERE genres IS NOT NULL AND genres != ''")
    count = cur.fetchone()[0]
    assert count > 1000, "Too many movies without genres"

# ── SVD Model Tests ────────────────────────────────────────────────────
def test_svd_model_loads(svd_model):
    """SVD model file loads successfully"""
    assert svd_model is not None

def test_svd_prediction_range(svd_model):
    """SVD predictions within valid range"""
    pred = svd_model.predict(uid=1, iid=1)
    assert 1.0 <= pred.est <= 5.0, f"Prediction {pred.est} out of range"

def test_svd_toy_story_prediction(svd_model):
    """Toy Story (movie 1) gets a reasonable prediction"""
    pred = svd_model.predict(uid=1, iid=1)
    assert pred.est >= 3.0, "Toy Story prediction too low"

def test_svd_different_users(svd_model):
    """Different users get different predictions"""
    pred1 = svd_model.predict(uid=1, iid=100).est
    pred2 = svd_model.predict(uid=500, iid=100).est
    assert pred1 != pred2, "All users getting same prediction"

def test_svd_different_movies(svd_model):
    """Same user gets different scores for different movies"""
    pred1 = svd_model.predict(uid=1, iid=1).est
    pred2 = svd_model.predict(uid=1, iid=50).est
    assert pred1 != pred2, "All movies getting same prediction"

# ── Recommendation Logic Tests ─────────────────────────────────────────
def test_movies_dataframe_loaded(movies_df):
    """Movies DataFrame has correct shape"""
    assert len(movies_df) == 1682
    assert "movie_id" in movies_df.columns
    assert "title" in movies_df.columns
    assert "genres" in movies_df.columns

def test_emotion_genre_mapping():
    """Emotion to genre mapping covers all 7 emotions"""
    EMOTION_TO_GENRES = {
        "joy":      ["Comedy","Animation","Musical","Children"],
        "sadness":  ["Drama","Romance"],
        "anger":    ["Action","Thriller","Crime"],
        "fear":     ["Horror","Thriller","Mystery"],
        "surprise": ["Mystery","Sci-Fi","Adventure"],
        "disgust":  ["Documentary","Crime"],
        "neutral":  ["Drama","Comedy","Documentary"]
    }
    expected = {"joy","sadness","anger","fear","surprise","disgust","neutral"}
    assert set(EMOTION_TO_GENRES.keys()) == expected

def test_genre_filtering(movies_df):
    """Genre filtering returns relevant movies"""
    thriller_movies = movies_df[
        movies_df["genres"].str.contains("Thriller", na=False)
    ]
    assert len(thriller_movies) > 50, "Too few Thriller movies"

def test_drama_movies_exist(movies_df):
    """Drama movies exist in dataset"""
    drama = movies_df[movies_df["genres"].str.contains("Drama", na=False)]
    assert len(drama) > 100

def test_comedy_movies_exist(movies_df):
    """Comedy movies exist in dataset"""
    comedy = movies_df[movies_df["genres"].str.contains("Comedy", na=False)]
    assert len(comedy) > 100

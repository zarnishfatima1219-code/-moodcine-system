"""
MoodCine — Unit Tests: TMDB API
================================
Run: python -m pytest tests/test_tmdb.py -v
"""
import requests
import pytest
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def test_api_key_exists():
    """Test TMDB API key is in .env file"""
    assert API_KEY is not None, "TMDB_API_KEY not found in .env"
    assert len(API_KEY) > 10, "API key seems too short"

def test_tmdb_connection():
    """Test TMDB API responds"""
    r = requests.get(f"{BASE_URL}/configuration",
                     params={"api_key": API_KEY}, timeout=10)
    assert r.status_code == 200, "TMDB API not reachable"

def test_toy_story_found():
    """Test a known movie can be found"""
    r = requests.get(f"{BASE_URL}/search/movie",
                     params={"api_key": API_KEY, "query": "Toy Story", "year": 1995},
                     timeout=10)
    data = r.json()
    assert data["total_results"] > 0, "Toy Story not found"
    assert data["results"][0]["title"] == "Toy Story"

def test_movie_has_poster():
    """Test fetched movie has poster URL"""
    r = requests.get(f"{BASE_URL}/search/movie",
                     params={"api_key": API_KEY, "query": "Inception"},
                     timeout=10)
    result = r.json()["results"][0]
    assert result.get("poster_path") is not None, "No poster found"

def test_movie_has_overview():
    """Test fetched movie has overview text"""
    r = requests.get(f"{BASE_URL}/search/movie",
                     params={"api_key": API_KEY, "query": "The Dark Knight"},
                     timeout=10)
    result = r.json()["results"][0]
    assert len(result.get("overview","")) > 20, "Overview too short or missing"

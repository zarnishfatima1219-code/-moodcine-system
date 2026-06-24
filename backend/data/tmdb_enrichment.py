"""
MoodCine — TMDB Data Enrichment Script
=======================================
Fetches poster, overview, cast, genres for MovieLens movies from TMDB API
Run: python backend/data/tmdb_enrichment.py
"""

import requests
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

DATASET_DIR = "dataset"
ML_DIR = os.path.join(DATASET_DIR, "ml-100k")
OUTPUT_DIR = os.path.join(DATASET_DIR, "enriched")
os.makedirs(OUTPUT_DIR, exist_ok=True)

genre_cols = ["unknown","Action","Adventure","Animation","Children","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

movies = pd.read_csv(os.path.join(ML_DIR, "u.item"), sep="|",
    encoding="latin-1",
    names=["movie_id","title","release_date","video_release","imdb_url"]+genre_cols,
    usecols=["movie_id","title","release_date"])

def clean_title(title):
    """Remove year from title like 'Toy Story (1995)' → 'Toy Story'"""
    import re
    return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

def get_year(title):
    """Extract year from title like 'Toy Story (1995)' → 1995"""
    import re
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else None

def search_movie(title, year=None):
    """Search TMDB for a movie by title and year"""
    params = {"api_key": API_KEY, "query": title, "language": "en-US"}
    if year:
        params["year"] = year
    try:
        r = requests.get(f"{BASE_URL}/search/movie", params=params, timeout=10)
        data = r.json()
        if data.get("results"):
            return data["results"][0]
    except Exception as e:
        print(f"   ⚠️  Search error: {e}")
    return None

def get_cast(tmdb_id):
    """Get top 3 cast members for a movie"""
    try:
        r = requests.get(f"{BASE_URL}/movie/{tmdb_id}/credits",
                        params={"api_key": API_KEY}, timeout=10)
        data = r.json()
        cast = data.get("cast", [])[:3]
        return [c["name"] for c in cast]
    except:
        return []

# ── Main enrichment loop ───────────────────────────────────────────────
print("="*60)
print("  MoodCine — TMDB Enrichment Script")
print("="*60)
print(f"\n📡 API Key loaded: {API_KEY[:8]}...")
print(f"🎬 Total movies to process: {len(movies)}")
print("\n⚡ Testing first 20 movies (full run = 1682 movies)")
print("   Full run takes ~30 mins due to API rate limits\n")

enriched = []
TEST_LIMIT = 20  # Change to len(movies) for full run

for idx, row in movies.head(TEST_LIMIT).iterrows():
    movie_id = row["movie_id"]
    title = row["title"]
    clean = clean_title(title)
    year = get_year(title)

    print(f"[{movie_id:4d}] Searching: {clean} ({year})...")

    result = search_movie(clean, year)

    if result:
        tmdb_id = result["id"]
        cast = get_cast(tmdb_id)
        poster_path = result.get("poster_path", "")
        poster_url = f"{IMG_BASE}{poster_path}" if poster_path else ""

        movie_data = {
            "movie_id": movie_id,
            "title": title,
            "tmdb_id": tmdb_id,
            "tmdb_title": result.get("title", ""),
            "overview": result.get("overview", "")[:200],
            "release_date": result.get("release_date", ""),
            "vote_average": result.get("vote_average", 0),
            "vote_count": result.get("vote_count", 0),
            "popularity": result.get("popularity", 0),
            "poster_url": poster_url,
            "cast": ", ".join(cast),
            "tmdb_genres": ", ".join([str(g) for g in result.get("genre_ids", [])]),
        }
        enriched.append(movie_data)
        print(f"       ✅ Found: {result.get('title')} | ⭐ {result.get('vote_average')}/10 | 🎭 {', '.join(cast[:2])}")
    else:
        enriched.append({
            "movie_id": movie_id, "title": title,
            "tmdb_id": None, "tmdb_title": "", "overview": "",
            "release_date": "", "vote_average": 0, "vote_count": 0,
            "popularity": 0, "poster_url": "", "cast": "", "tmdb_genres": ""
        })
        print(f"       ❌ Not found on TMDB")

    time.sleep(0.3)  # Rate limit: max ~40 requests/second

# Save results
df = pd.DataFrame(enriched)
out_path = os.path.join(OUTPUT_DIR, "movies_enriched.csv")
df.to_csv(out_path, index=False)

print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
found = df[df["tmdb_id"].notna()]
print(f"\n  ✅ Movies found on TMDB : {len(found)}/{TEST_LIMIT}")
print(f"  ❌ Not found           : {TEST_LIMIT - len(found)}/{TEST_LIMIT}")
print(f"  📊 Average TMDB rating : {found['vote_average'].mean():.1f}/10")
print(f"\n  💾 Saved → {out_path}")
print("\n  Sample data:")
print(df[["movie_id","title","vote_average","cast"]].head(5).to_string(index=False))
print("\n📸 SCREENSHOT THIS FOR EVIDENCE!")

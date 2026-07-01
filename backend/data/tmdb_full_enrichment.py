"""
MoodCine — TMDB Full Enrichment Script
=======================================
Enriches ALL 1682 MovieLens movies with TMDB data
Run: python backend/data/tmdb_full_enrichment.py

Time: ~20-25 minutes (API rate limiting)
"""
import requests
import pandas as pd
import psycopg2
import json
import os
import time
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

DB = {"host":"localhost","database":"moodcine","user":"postgres",
      "password":"moodcine123","port":5432}

ML_DIR      = "dataset/ml-100k"
OUTPUT_DIR  = "dataset/enriched"
os.makedirs(OUTPUT_DIR, exist_ok=True)

genre_cols = ["unknown","Action","Adventure","Animation","Children","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

movies = pd.read_csv(f"{ML_DIR}/u.item", sep="|", encoding="latin-1",
    names=["movie_id","title","release_date","video","imdb"]+genre_cols,
    usecols=["movie_id","title","release_date"])

def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)\s*$','',title).strip()

def get_year(title):
    m = re.search(r'\((\d{4})\)',title)
    return m.group(1) if m else None

def search_movie(title, year=None):
    params = {"api_key":API_KEY,"query":title,"language":"en-US"}
    if year: params["year"] = year
    try:
        r = requests.get(f"{BASE_URL}/search/movie",params=params,timeout=10)
        data = r.json()
        if data.get("results"):
            return data["results"][0]
    except Exception as e:
        print(f"   Search error: {e}")
    return None

def get_cast(tmdb_id):
    try:
        r = requests.get(f"{BASE_URL}/movie/{tmdb_id}/credits",
                        params={"api_key":API_KEY},timeout=10)
        cast = r.json().get("cast",[])[:3]
        return [c["name"] for c in cast]
    except:
        return []

def update_db(conn, movie_id, data):
    cur = conn.cursor()
    cur.execute("""
        UPDATE movies SET
            tmdb_id      = %s,
            overview     = %s,
            vote_average = %s,
            poster_url   = %s,
            movie_cast   = %s
        WHERE movie_id = %s
    """, (data["tmdb_id"], data["overview"], data["vote_average"],
          data["poster_url"], data["cast"], movie_id))
    conn.commit()

# ── MAIN ──────────────────────────────────────────────────────────────
print("="*60)
print("  MoodCine — TMDB Full Enrichment (All 1682 Movies)")
print("="*60)
print(f"\n  API Key : {API_KEY[:8]}...")
print(f"  Movies  : {len(movies)}")
print(f"  Est time: ~20-25 minutes\n")

conn = psycopg2.connect(**DB)
print("Connected to PostgreSQL!\n")

enriched   = []
found      = 0
not_found  = 0
errors     = 0

start_time = time.time()

for idx, row in movies.iterrows():
    movie_id = int(row["movie_id"])
    title    = row["title"]
    clean    = clean_title(title)
    year     = get_year(title)
    num      = movie_id

    result = search_movie(clean, year)

    if result:
        tmdb_id     = result["id"]
        cast        = get_cast(tmdb_id)
        poster_path = result.get("poster_path","")
        poster_url  = f"{IMG_BASE}{poster_path}" if poster_path else ""

        data = {
            "movie_id"    : movie_id,
            "title"       : title,
            "tmdb_id"     : tmdb_id,
            "tmdb_title"  : result.get("title",""),
            "overview"    : result.get("overview","")[:300],
            "release_date": result.get("release_date",""),
            "vote_average": result.get("vote_average",0),
            "vote_count"  : result.get("vote_count",0),
            "poster_url"  : poster_url,
            "cast"        : ", ".join(cast),
        }
        enriched.append(data)
        update_db(conn, movie_id, data)
        found += 1

        if movie_id % 50 == 0 or movie_id <= 20:
            elapsed = time.time() - start_time
            rate = found / elapsed if elapsed > 0 else 0
            remaining = (len(movies) - num) / rate if rate > 0 else 0
            print(f"  [{num:4d}/{len(movies)}] ✅ {title[:35]:<35} "
                  f"⭐{result.get('vote_average',0):.1f} "
                  f"| ~{remaining/60:.0f} min left")
    else:
        enriched.append({
            "movie_id":movie_id,"title":title,"tmdb_id":None,
            "tmdb_title":"","overview":"","release_date":"",
            "vote_average":0,"vote_count":0,"poster_url":"","cast":""
        })
        not_found += 1
        if not_found <= 10:
            print(f"  [{num:4d}/{len(movies)}] ❌ Not found: {title[:40]}")

    time.sleep(0.25)

conn.close()

# Save CSV
df = pd.DataFrame(enriched)
out = f"{OUTPUT_DIR}/movies_enriched_full.csv"
df.to_csv(out, index=False)

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"  ENRICHMENT COMPLETE!")
print(f"{'='*60}")
print(f"  Total movies    : {len(movies)}")
print(f"  Found on TMDB   : {found} ({found/len(movies)*100:.1f}%)")
print(f"  Not found       : {not_found}")
print(f"  Time taken      : {elapsed_total/60:.1f} minutes")
print(f"  Output CSV      : {out}")
print(f"\n  Top 5 rated movies found:")
top5 = df[df['vote_average']>0].nlargest(5,'vote_average')[['title','vote_average']]
print(top5.to_string(index=False))
print(f"\nDatabase updated — PostgreSQL movies table enriched!")
print(f"SCREENSHOT THIS FOR EVIDENCE!")

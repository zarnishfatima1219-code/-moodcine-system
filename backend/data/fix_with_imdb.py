"""
MoodCine — Fix Not Found Movies
Tries alternative title formats for better TMDB matching
"""
import requests
import pandas as pd
import psycopg2
import re
import time
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY  = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
DB = {"host":"localhost","database":"moodcine","user":"postgres","password":"moodcine123","port":5432}

def fix_title(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$','',title).strip()
    for article in [', The',', A',', An',', Les',', La',', Le',', Der',', Das']:
        if clean.endswith(article):
            word = article.replace(', ','')
            clean = word+' '+clean[:-len(article)]
            break
    return clean

def search_tmdb(query, year=None):
    try:
        params = {"api_key":API_KEY,"query":query,"language":"en-US"}
        if year: params["year"] = year
        r = requests.get(f"{BASE_URL}/search/movie",params=params,timeout=10)
        results = r.json().get("results",[])
        return results[0] if results else None
    except:
        return None

def get_cast(tmdb_id):
    try:
        r = requests.get(f"{BASE_URL}/movie/{tmdb_id}/credits",
                        params={"api_key":API_KEY},timeout=10)
        return ", ".join([c["name"] for c in r.json().get("cast",[])[:3]])
    except:
        return ""

conn = psycopg2.connect(**DB)
cur  = conn.cursor()
cur.execute("SELECT movie_id, title FROM movies WHERE tmdb_id IS NULL ORDER BY movie_id")
missing = cur.fetchall()

print(f"Movies without TMDB data: {len(missing)}")
print("Trying alternative title formats...\n")

fixed = 0
still_missing = []

for movie_id, title in missing:
    year_match = re.search(r'\((\d{4})\)',title)
    year = year_match.group(1) if year_match else None
    fixed_t = fix_title(title)
    clean_t  = re.sub(r'\s*\(\d{4}\)\s*$','',title).strip()

    result = (search_tmdb(fixed_t, year) or
              search_tmdb(fixed_t) or
              search_tmdb(clean_t, year) or
              search_tmdb(clean_t))

    if result:
        cast = get_cast(result["id"])
        pp   = result.get("poster_path","")
        cur.execute("""
            UPDATE movies SET tmdb_id=(%s),overview=(%s),
            vote_average=(%s),poster_url=(%s),movie_cast=(%s)
            WHERE movie_id=(%s)
        """, (result["id"],result.get("overview","")[:300],
              result.get("vote_average",0),
              f"{IMG_BASE}{pp}" if pp else "",cast,movie_id))
        conn.commit()
        fixed += 1
        print(f"  ✅ Fixed [{movie_id:4d}]: {title[:40]:<40} → {result.get('title','')[:25]}")
    else:
        still_missing.append((movie_id,title))
    time.sleep(0.25)

conn.close()
print(f"\n{'='*55}")
print(f"  Fixed     : {fixed}")
print(f"  Still missing: {len(still_missing)}")
if still_missing:
    print(f"\n  Still not found:")
    for mid,t in still_missing:
        print(f"    [{mid}] {t}")
print(f"\nSCREENSHOT THIS!")

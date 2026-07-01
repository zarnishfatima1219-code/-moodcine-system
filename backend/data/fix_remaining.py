"""
MoodCine — Fix Remaining 14 Movies
Manual alternative titles for very hard to find movies
"""
import requests
import psycopg2
import time
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY  = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

DB = {"host":"localhost","database":"moodcine","user":"postgres",
      "password":"moodcine123","port":5432}

# Manual alternatives for hard-to-find movies
MANUAL = {
    563:  [("The Langoliers", 1995), ("Langoliers", None)],
    912:  [("U.S. Marshals", 1998), ("US Marshals", 1998)],
    1201: [("Marlene Dietrich Shadow and Light", None), ("Marlene", 1996)],
    1235: [("The Big Bang", 1994), ("Big Bang", 1994)],
    1345: [("The Day the Sun Turned Cold", 1994), ("Tianguo niezi", 1994)],
    1359: [("Boys in Venice", 1996), ("Venice", 1996)],
    1405: [("Boys Life 2", 1997), ("Boy's Life 2", None)],
    1420: [("Gilligan's Island", 1998), ("Gilligan Island Movie", None)],
    1536: [("Vive L'Amour", 1994), ("Aiqing wansui", None)],
    1569: [("La Vie est belle", 1987), ("Life is Rosy", 1987)],
    1591: [("Fallen Angels", 1995), ("Duo luo tian shi", None)],
    1630: [("The Silences of the Palace", 1994), ("Silence of the Palace", None)],
    1634: [("Under the Domim Tree", 1994), ("Etz Hadomim Tafus", None)],
    1639: [("Azucar Amargo", 1996), ("Bitter Sugar", 1996)],
}

def search(query, year=None):
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

print("="*55)
print("  MoodCine — Fixing Remaining 14 Movies")
print("="*55)

fixed = 0
not_fixed = []

for movie_id, alternatives in MANUAL.items():
    cur.execute("SELECT title FROM movies WHERE movie_id=%s",(movie_id,))
    row = cur.fetchone()
    if not row: continue
    title = row[0]

    result = None
    for query, year in alternatives:
        result = search(query, year)
        if result: break
        time.sleep(0.2)

    if result:
        cast = get_cast(result["id"])
        pp   = result.get("poster_path","")
        cur.execute("""
            UPDATE movies SET
                tmdb_id=(%s),overview=(%s),vote_average=(%s),
                poster_url=(%s),movie_cast=(%s)
            WHERE movie_id=(%s)
        """, (result["id"],result.get("overview","")[:300],
              result.get("vote_average",0),
              f"{IMG_BASE}{pp}" if pp else "",
              cast, movie_id))
        conn.commit()
        fixed += 1
        print(f"  ✅ [{movie_id:4d}] {title[:35]:<35} → {result.get('title','')[:25]}")
    else:
        not_fixed.append((movie_id, title))
        print(f"  ❌ [{movie_id:4d}] {title[:35]} — truly not on TMDB")

    time.sleep(0.3)

conn.close()

print(f"\n{'='*55}")
print(f"  Fixed    : {fixed}")
print(f"  Not fixed: {len(not_fixed)}")

# Final count
conn2 = psycopg2.connect(**DB)
cur2  = conn2.cursor()
cur2.execute("SELECT COUNT(*) FROM movies WHERE poster_url != '' AND poster_url IS NOT NULL")
total = cur2.fetchone()[0]
conn2.close()

print(f"\n  Total movies with posters: {total}/1682")
print(f"  Coverage: {total/1682*100:.1f}%")
print(f"\nSCREENSHOT THIS!")

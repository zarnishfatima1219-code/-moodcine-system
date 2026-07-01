"""
MoodCine — Test Poster URLs
Verifies that poster URLs actually return images
"""
import psycopg2
import requests

DB = {"host":"localhost","database":"moodcine","user":"postgres",
      "password":"moodcine123","port":5432}

conn = psycopg2.connect(**DB)
cur  = conn.cursor()

# Get 10 random movies with posters
cur.execute("""
    SELECT movie_id, title, poster_url, vote_average, movie_cast
    FROM movies 
    WHERE poster_url != '' AND poster_url IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 10
""")
movies = cur.fetchall()
conn.close()

print("="*65)
print("  MoodCine — Poster URL Test (10 random movies)")
print("="*65)

working = 0
broken  = 0

for movie_id, title, poster_url, rating, cast in movies:
    try:
        r = requests.head(poster_url, timeout=8)
        if r.status_code == 200:
            working += 1
            print(f"  ✅ [{movie_id:4d}] {title[:35]:<35} ⭐{rating:.1f}")
            print(f"         Cast: {cast[:50] if cast else 'N/A'}")
            print(f"         URL : {poster_url[:60]}...")
        else:
            broken += 1
            print(f"  ❌ [{movie_id:4d}] {title[:35]:<35} (HTTP {r.status_code})")
    except Exception as e:
        broken += 1
        print(f"  ⚠️  [{movie_id:4d}] {title[:35]:<35} Error: {e}")
    print()

print("="*65)
print(f"  Working posters : {working}/10")
print(f"  Broken posters  : {broken}/10")
print("="*65)
print("📸 SCREENSHOT THIS FOR EVIDENCE!")

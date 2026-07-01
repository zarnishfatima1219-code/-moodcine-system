"""
MoodCine — Data Loading Script
Loads MovieLens 100K into PostgreSQL
"""
import pandas as pd
import psycopg2

DB = {
    "host": "localhost",
    "database": "moodcine",
    "user": "postgres",
    "password": "moodcine123",
    "port": 5432
}

ML_DIR = "dataset/ml-100k"

genre_cols = ["unknown","Action","Adventure","Animation","Children",
    "Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir",
    "Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def load_movies(conn):
    print("🎬 Loading movies...")
    movies = pd.read_csv(f"{ML_DIR}/u.item", sep="|",
        encoding="latin-1",
        names=["movie_id","title","release_date","video","imdb"]+genre_cols,
        usecols=["movie_id","title","release_date"]+genre_cols)
    cur = conn.cursor()
    count = 0
    for _, row in movies.iterrows():
        genre_list = [g for g in genre_cols if row[g] == 1]
        genres_str = ", ".join(genre_list)
        cur.execute("""
            INSERT INTO movies (movie_id, title, genres, release_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (movie_id) DO NOTHING
        """, (int(row["movie_id"]), row["title"], genres_str, str(row["release_date"])))
        count += 1
    conn.commit()
    print(f"✅ {count} movies loaded!")
    return count

def load_users_and_ratings(conn):
    print("⭐ Loading ratings...")
    ratings = pd.read_csv(f"{ML_DIR}/u.data", sep="\t",
        names=["user_id","movie_id","rating","timestamp"])
    cur = conn.cursor()
    unique_users = ratings["user_id"].unique()
    for uid in unique_users:
        cur.execute("""
            INSERT INTO users (user_id, username, email, password_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (int(uid), f"user_{uid}", f"user_{uid}@moodcine.com", "placeholder"))
    conn.commit()
    print(f"✅ {len(unique_users)} users loaded!")
    count = 0
    for _, row in ratings.iterrows():
        cur.execute("""
            INSERT INTO ratings (user_id, movie_id, rating)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (int(row["user_id"]), int(row["movie_id"]), int(row["rating"])))
        count += 1
        if count % 10000 == 0:
            conn.commit()
            print(f"   {count:,} ratings done...")
    conn.commit()
    print(f"✅ {count:,} ratings loaded!")

def main():
    print("="*55)
    print("  MoodCine — Loading Data into PostgreSQL")
    print("="*55)
    conn = psycopg2.connect(**DB)
    print("✅ Connected!\n")
    load_movies(conn)
    load_users_and_ratings(conn)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM movies")
    print(f"\n📊 Movies in DB  : {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM ratings")
    print(f"📊 Ratings in DB : {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM users")
    print(f"📊 Users in DB   : {cur.fetchone()[0]:,}")
    conn.close()
    print("\n✅ DONE! Screenshot this!")

if __name__ == "__main__":
    main()

import os, zipfile, requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "dataset"
ML_DIR = os.path.join(DATASET_DIR, "ml-100k")
EVIDENCE_DIR = "screenshots_evidence"
os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATASET_DIR, "ml-100k.zip")

if not os.path.exists(os.path.join(ML_DIR, "u.data")):
    print("Downloading MovieLens 100K...")
    r = requests.get(ZIP_URL, stream=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(8192): f.write(chunk)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATASET_DIR)
    print("Downloaded!\n")
else:
    print("Already downloaded.\n")

ratings = pd.read_csv(os.path.join(ML_DIR, "u.data"), sep="\t",
    names=["user_id","movie_id","rating","timestamp"])

genre_cols = ["unknown","Action","Adventure","Animation","Children","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

movies = pd.read_csv(os.path.join(ML_DIR, "u.item"), sep="|", encoding="latin-1",
    names=["movie_id","title","release_date","video_release","imdb_url"]+genre_cols,
    usecols=["movie_id","title","release_date"]+genre_cols)

df = ratings.merge(movies[["movie_id","title"]+genre_cols], on="movie_id", how="left")

n_users  = df["user_id"].nunique()
n_movies = df["movie_id"].nunique()
avg_rating = df["rating"].mean()
sparsity = 1 - (len(df) / (n_users * n_movies))

print("="*55)
print("   MoodCine - MovieLens 100K Loaded!")
print("="*55)
print(f"  Total ratings  : {len(df):,}")
print(f"  Unique users   : {n_users:,}")
print(f"  Unique movies  : {n_movies:,}")
print(f"  Avg rating     : {avg_rating:.2f} / 5.0")
print(f"  Sparsity       : {sparsity:.1%}")
print("="*55)
print("\nSample rows:")
print(df[["user_id","movie_id","title","rating"]].head(8).to_string(index=False))
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MoodCine - MovieLens 100K Analysis", fontsize=13, fontweight="bold")
rc = df["rating"].value_counts().sort_index()
axes[0].bar(rc.index, rc.values, color=["#4e79a7","#59a14f","#f28e2b","#e15759","#76b7b2"])
axes[0].set_title("Rating Distribution")
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Count")
gc = movies[genre_cols].sum().sort_values(ascending=False).head(10)
axes[1].bar(gc.index, gc.values, color="#4e79a7")
axes[1].set_title("Top 10 Genres")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
out = os.path.join(EVIDENCE_DIR, "week1_dataset_chart.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved to: {out}")
print("TAKE A SCREENSHOT OF THIS TERMINAL FOR TEACHER")

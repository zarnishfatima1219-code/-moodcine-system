"""
MoodCine — SVD Collaborative Filtering Model
Trains Scikit-Surprise SVD on MovieLens 100K
"""
import psycopg2
import pandas as pd
import pickle
import os
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

DB = {"host":"localhost","database":"moodcine","user":"postgres","password":"moodcine123","port":5432}

print("="*55)
print("  MoodCine — Training SVD Model")
print("="*55)

# Step 1: Load ratings from database
print("\n📥 Loading ratings from PostgreSQL...")
conn = psycopg2.connect(**DB)
df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
conn.close()
print(f"✅ {len(df):,} ratings loaded from DB!")

# Step 2: Prepare data for Surprise
print("\n🔧 Preparing data for SVD...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id","movie_id","rating"]], reader)

# Step 3: Train/test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print(f"✅ Train: {trainset.n_ratings:,} ratings")
print(f"✅ Test : {len(testset):,} ratings")

# Step 4: Train SVD model
print("\n🤖 Training SVD model...")
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)
print("✅ Model trained!")

# Step 5: Evaluate
print("\n📊 Evaluating model...")
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae  = accuracy.mae(predictions)
print(f"✅ RMSE : {rmse:.4f}  (lower is better)")
print(f"✅ MAE  : {mae:.4f}   (lower is better)")

# Step 6: Save model
os.makedirs("backend/ml/recommender", exist_ok=True)
model_path = "backend/ml/recommender/svd_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\n💾 Model saved → {model_path}")

# Step 7: Quick test
print("\n🎬 Quick test — Toy Story (movie_id=1) for user_1:")
prediction = model.predict(uid=1, iid=1)
print(f"   Predicted rating: {prediction.est:.2f}/5.0")

print("\n" + "="*55)
print("  SVD MODEL TRAINING COMPLETE!")
print("="*55)
print("📸 SCREENSHOT THIS FOR EVIDENCE!")

from transformers import pipeline
import json, os

EMOTION_TO_FILM = {
    "joy":      {"genres": ["Comedy","Animation","Musical"],    "tone": "uplifting",           "pacing": "fast"},
    "sadness":  {"genres": ["Drama","Romance"],                 "tone": "emotional, cathartic","pacing": "slow"},
    "anger":    {"genres": ["Action","Thriller","Crime"],       "tone": "intense",             "pacing": "fast"},
    "fear":     {"genres": ["Horror","Thriller","Mystery"],     "tone": "tense, suspenseful",  "pacing": "moderate"},
    "surprise": {"genres": ["Mystery","Sci-Fi","Adventure"],    "tone": "mind-bending",        "pacing": "fast"},
    "disgust":  {"genres": ["Documentary","Crime"],             "tone": "gritty, realistic",   "pacing": "moderate"},
    "neutral":  {"genres": ["Documentary","Drama","Comedy"],    "tone": "balanced",            "pacing": "moderate"},
}

TEST_INPUTS = [
    "I feel really anxious and mentally exhausted after a long week",
    "I am so happy and excited, everything is going great today!",
    "I am furious about what happened, I cannot calm down",
    "I feel quite sad and lonely, nothing seems right",
    "I want something totally unexpected and mind-blowing tonight",
]

print("="*60)
print("  MoodCine - NLP Emotion Detection Test")
print("  Model: DistilRoBERTa (j-hartmann)")
print("="*60)
print("\nLoading model...\n")

clf = pipeline("text-classification",
               model="j-hartmann/emotion-english-distilroberta-base",
               return_all_scores=True)
print("✅ Real model loaded!\n")

def get_scores(text):
    result = clf(text)
    # Fix for newer transformers versions
    if isinstance(result[0], list):
        raw = result[0]
    elif isinstance(result[0], dict):
        raw = result
    else:
        raw = result[0]
    return sorted(raw, key=lambda x: x["score"], reverse=True)

results = []
for i, text in enumerate(TEST_INPUTS):
    print(f"{'─'*60}")
    print(f" Test {i+1}: \"{text}\"")
    scores = get_scores(text)
    top = scores[0]["label"]
    attrs = EMOTION_TO_FILM.get(top, {})
    print("\n  Emotion Scores:")
    for s in scores:
        bar = "█" * int(s["score"]*28) + "░" * (28 - int(s["score"]*28))
        mark = " ◀ TOP" if s["label"] == top else ""
        print(f"    {s['label']:<10} {bar} {s['score']:.3f}{mark}")
    print(f"\n  🎯 Emotion : {top.upper()} ({scores[0]['score']:.1%} confidence)")
    print(f"  🎬 Genres  : {', '.join(attrs.get('genres', []))}")
    print(f"  🎨 Tone    : {attrs.get('tone', '')}\n")
    results.append({"text": text, "emotion": top,
                    "confidence": round(scores[0]["score"], 3),
                    "genres": attrs.get("genres", [])})

print("="*60)
print("  SUMMARY TABLE")
print("="*60)
for i, r in enumerate(results, 1):
    print(f"  {i}. {r['emotion'].upper():<10} {r['confidence']:.0%}  ->  {', '.join(r['genres'][:2])}")

os.makedirs("screenshots_evidence", exist_ok=True)
with open("screenshots_evidence/week1_nlp_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✅ Results saved to screenshots_evidence/week1_nlp_results.json")
print("📸 SCREENSHOT THIS FOR YOUR TEACHER!")

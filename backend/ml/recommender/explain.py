"""
MoodCine — Gemini Explanation Generator
"""
from google import genai
import os
import time
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_explanation(emotion, movie_title, genre, overview):
    prompt = f"""You are MoodCine, an emotion-aware movie recommendation system.
A user is feeling: {emotion}
You are recommending: {movie_title} (Genre: {genre})
Movie overview: {overview[:150]}
In exactly 2 sentences, explain why this movie suits someone feeling {emotion} right now.
Be warm, empathetic and specific."""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            time.sleep(6)
            try:
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt
                )
                return response.text.strip()
            except:
                return f"A perfect choice for your {emotion} mood."
        return f"A great pick for when you are feeling {emotion}."

if __name__ == "__main__":
    print("="*60)
    print("  MoodCine — Gemini Explanation Test")
    print("="*60)

    tests = [
        ("fear",    "The Silence of the Lambs", "Thriller", "Clarice interviews Hannibal Lecter to catch a serial killer."),
        ("sadness", "Forrest Gump",             "Drama",    "A man with low IQ achieves great things in life."),
        ("joy",     "Toy Story",                "Animation","A cowboy doll is threatened by a new spaceman toy."),
    ]

    for emotion, title, genre, overview in tests:
        print(f"\n  Emotion : {emotion.upper()}")
        print(f"  Movie   : {title}")
        exp = generate_explanation(emotion, title, genre, overview)
        print(f"  AI Says : {exp}")
        print(f"  {'-'*55}")
        time.sleep(4)

    print(f"\n{'='*60}")
    print("SCREENSHOT THIS FOR EVIDENCE!")

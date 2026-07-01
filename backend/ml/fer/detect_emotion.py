"""
MoodCine — FER Emotion Detection Module
========================================
Uses DeepFace to detect emotion from webcam frame
Run: py -3.12 backend/ml/fer/detect_emotion.py
"""
import cv2
import numpy as np
from deepface import DeepFace
import json

EMOTION_MAP = {
    "angry"  : "anger",
    "disgust": "disgust",
    "fear"   : "fear",
    "happy"  : "joy",
    "sad"    : "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}

def detect_from_frame(frame):
    """
    Detect emotion from a single webcam frame
    Returns: dict with emotion scores
    """
    try:
        result = DeepFace.analyze(
            img_path    = frame,
            actions     = ["emotion"],
            enforce_detection = False,
            silent      = True
        )
        raw_emotions = result[0]["emotion"]

        # Map DeepFace labels to MoodCine labels
        mapped = {}
        for deepface_label, score in raw_emotions.items():
            moodcine_label = EMOTION_MAP.get(deepface_label, deepface_label)
            mapped[moodcine_label] = round(score / 100, 4)

        dominant = result[0]["dominant_emotion"]
        dominant_mapped = EMOTION_MAP.get(dominant, dominant)

        return {
            "success"         : True,
            "dominant_emotion": dominant_mapped,
            "confidence"      : round(mapped.get(dominant_mapped, 0) * 100, 1),
            "all_scores"      : mapped
        }
    except Exception as e:
        return {
            "success"         : False,
            "error"           : str(e),
            "dominant_emotion": "neutral",
            "confidence"      : 0.0,
            "all_scores"      : {}
        }

def detect_from_webcam():
    """
    Capture single frame from webcam and detect emotion
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"success": False, "error": "Webcam not accessible"}

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"success": False, "error": "Could not read frame"}

    return detect_from_frame(frame)

def test_with_image():
    """
    Test FER with a sample image (no webcam needed)
    """
    import urllib.request
    import os

    test_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/300px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    test_img_path = "screenshots_evidence/week4/test_face.jpg"
    os.makedirs("screenshots_evidence/week4", exist_ok=True)

    print("Downloading test image...")
    urllib.request.urlretrieve(test_img_url, test_img_path)
    print("Running DeepFace emotion detection...")

    img = cv2.imread(test_img_path)
    result = detect_from_frame(img)
    return result

if __name__ == "__main__":
    print("="*55)
    print("  MoodCine — FER Emotion Detection Test")
    print("="*55)
    print("\nTesting with sample image (no webcam needed)...\n")

    result = test_with_image()

    if result["success"]:
        print(f"  Dominant Emotion : {result['dominant_emotion'].upper()}")
        print(f"  Confidence       : {result['confidence']}%")
        print(f"\n  All Emotion Scores:")
        for emotion, score in sorted(result['all_scores'].items(),
                                     key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"    {emotion:<12} {bar:<20} {score*100:.1f}%")
    else:
        print(f"  Error: {result['error']}")

    print(f"\n{'='*55}")
    print("SCREENSHOT THIS FOR EVIDENCE!")

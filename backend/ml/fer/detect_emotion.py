"""
MoodCine — FER Emotion Detection Module
========================================
Uses DeepFace for webcam + image emotion detection
Run: py -3.12 backend/ml/fer/detect_emotion.py
"""
import cv2
from deepface import DeepFace

EMOTION_MAP = {
    "angry"   : "anger",
    "disgust" : "disgust",
    "fear"    : "fear",
    "happy"   : "joy",
    "sad"     : "sadness",
    "surprise": "surprise",
    "neutral" : "neutral"
}

def detect_from_frame(frame):
    """Detect emotion from image/frame"""
    try:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )
        raw = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"]
        mapped_scores = {EMOTION_MAP.get(k,k): round(v/100,4) for k,v in raw.items()}
        dominant_mapped = EMOTION_MAP.get(dominant, dominant)
        return {
            "success"         : True,
            "dominant_emotion": dominant_mapped,
            "confidence"      : round(mapped_scores[dominant_mapped]*100, 1),
            "all_scores"      : mapped_scores,
            "source"          : "image"
        }
    except Exception as e:
        return {"success":False,"error":str(e),"dominant_emotion":"neutral","confidence":0.0,"all_scores":{}}

def detect_from_webcam():
    """
    Capture frame from webcam and detect emotion
    Called by FastAPI /fer-emotion endpoint
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"success":False,"error":"Webcam not accessible","dominant_emotion":"neutral","confidence":0.0}

        # Warmup frames
        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"success":False,"error":"Could not read frame","dominant_emotion":"neutral","confidence":0.0}

        result = detect_from_frame(frame)
        result["source"] = "webcam"
        return result

    except Exception as e:
        return {"success":False,"error":str(e),"dominant_emotion":"neutral","confidence":0.0}

def detect_from_base64(base64_str):
    """
    Detect emotion from base64 image
    Called by React frontend via FastAPI
    """
    import base64
    import numpy as np
    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = detect_from_frame(frame)
        result["source"] = "base64"
        return result
    except Exception as e:
        return {"success":False,"error":str(e),"dominant_emotion":"neutral","confidence":0.0}

if __name__ == "__main__":
    print("="*55)
    print("  MoodCine — FER Live Webcam Test")
    print("="*55)
    print("\nOpening webcam — please look at camera...\n")

    result = detect_from_webcam()

    if result["success"]:
        print(f"  Source           : {result['source'].upper()}")
        print(f"  Dominant Emotion : {result['dominant_emotion'].upper()}")
        print(f"  Confidence       : {result['confidence']}%")
        print(f"\n  All Emotion Scores:")
        for emotion, score in sorted(result['all_scores'].items(),
                                     key=lambda x:x[1], reverse=True):
            bar = chr(9608) * int(score*40)
            print(f"    {emotion:<12} {bar:<40} {score*100:.1f}%")
        print(f"\n  Webcam FER working!")
    else:
        print(f"  Error: {result['error']}")

    print(f"\n{'='*55}")
    print("SCREENSHOT THIS FOR EVIDENCE!")

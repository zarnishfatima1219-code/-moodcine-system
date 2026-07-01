"""
MoodCine — Emotion Fusion Layer
NLP (60%) + FER (40%) weighted combination
"""

EMOTIONS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]

def fuse_emotions(nlp_scores, fer_scores, nlp_weight=0.6, fer_weight=0.4):
    fused = {}
    for e in EMOTIONS:
        fused[e] = round((nlp_scores.get(e,0)*nlp_weight) + (fer_scores.get(e,0)*fer_weight), 4)
    total = sum(fused.values())
    if total > 0:
        fused = {k:round(v/total,4) for k,v in fused.items()}
    dominant = max(fused, key=fused.get)
    return {"fused_scores":fused,"dominant_emotion":dominant,"confidence":round(fused[dominant]*100,1)}

if __name__ == "__main__":
    print("="*55)
    print("  MoodCine — Fusion Layer Test")
    print("="*55)

    nlp = {"fear":0.993,"sadness":0.003,"anger":0.002,"joy":0.001,"neutral":0.001,"disgust":0.0,"surprise":0.0}
    fer = {"neutral":0.607,"joy":0.296,"sadness":0.080,"anger":0.013,"fear":0.004,"disgust":0.0,"surprise":0.0}

    result = fuse_emotions(nlp, fer)

    print(f"\n  NLP Input (60%): FEAR {nlp['fear']*100:.1f}%")
    print(f"  FER Input (40%): NEUTRAL {fer['neutral']*100:.1f}%")
    print(f"\n  FUSED RESULT:")
    print(f"  Dominant  : {result['dominant_emotion'].upper()}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"\n  All Fused Scores:")
    for e,s in sorted(result['fused_scores'].items(),key=lambda x:x[1],reverse=True):
        bar = chr(9608)*int(s*40)
        print(f"    {e:<12} {bar:<40} {s*100:.1f}%")
    print(f"\n{'='*55}")
    print("SCREENSHOT THIS FOR EVIDENCE!")

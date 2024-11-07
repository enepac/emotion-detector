import requests
import json

def emotion_detector(text_to_analyze):
    if not text_to_analyze.strip():  # Check for blank input
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
    
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    data = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200 or not response.text.strip():
        # Handle non-200 responses and empty responses gracefully
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    response_data = json.loads(response.text)
    emotions = response_data.get('emotionPredictions', [{}])[0].get('emotion', {})

    # Check if emotion data is present, otherwise default to None
    if not emotions:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Extract individual emotions and their scores
    emotion_scores = {emotion: emotions.get(emotion, 0) for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness']}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get, default=None)

    # Handle case where all scores are zero, indicating no meaningful data
    if all(score == 0 for score in emotion_scores.values()):
        dominant_emotion = None

    emotion_scores['dominant_emotion'] = dominant_emotion
    return emotion_scores

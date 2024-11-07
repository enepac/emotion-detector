"""
Flask application for emotion detection using Watson NLP.
Provides an endpoint to analyze emotions from input text.
"""

from flask import Flask, request, jsonify
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)

@app.route('/emotionDetector', methods=['POST'])
def detect_emotion():
    """
    Endpoint to detect emotions from the provided input text.
    
    Expects a JSON payload with a 'text' field containing the text to analyze.
    Returns a JSON response with emotion scores and a dominant emotion,
    or an error message if the input is invalid.
    """
    input_data = request.json
    text_to_analyze = input_data.get('text', '')

    # Run the emotion detection function
    result = emotion_detector(text_to_analyze)

    # Handle the case where dominant_emotion is None
    if result['dominant_emotion'] is None:
        return jsonify({
            "error": "Invalid text! Please try again!"
        })

    # Create response message
    response_message = (
        f"For the given statement, the system response is 'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, 'fear': {result['fear']}, "
        f"'joy': {result['joy']} and 'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )

    return jsonify({
        "emotions": result,
        "message": response_message
    })

if __name__ == '__main__':
    app.run(host='localhost', port=5000)

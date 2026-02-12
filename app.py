from flask import Flask, render_template, request, jsonify
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the emotion detection function from emotion_system.py
# We'll need to adapt the imports
import re
import random
import joblib
import difflib
import pandas as pd
import numpy as np
import collections
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
for pkg in ("stopwords", "wordnet"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
except Exception:
    class _DummySIA:
        def polarity_scores(self, text):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    sia = _DummySIA()

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

KEEP = {"not", "no", "so", "very", "really", "too"}
STOP_WORDS = STOP_WORDS.difference(KEEP)
lemmatizer = WordNetLemmatizer()

# Preprocessing
def preprocess_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    t = re.sub(r"[^\w\s'😀-🙏]", " ", t)
    tokens = t.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOP_WORDS]
    return " ".join(tokens)

# Emotion dictionary + emoji mapping
EMO_DICT = {
    "joy": {
        "happy","happiness","joy","joyful","glad","glee","delight","delighted","cheerful","cheer",
        "content","pleased","pleasure","bliss","ecstatic","elated","overjoyed","jubilant","smile","smiling",
        "thrilled","excited","yay","woohoo","win","winning","stoked","pumped","lit","lovely","love","cozy",
        "good","yaay","yaas","joyous","satisfied","satisfying","sunny","merry","cheery","sparkle"
    },
    "sadness": {
        "sad","sadness","sorrow","unhappy","miserable","depressed","depression","lonely","loneliness","cry",
        "crying","tears","grief","hopeless","despair","broken","empty","blue","melancholy","down","bored",
        "boring","bleh","meh","heartbroken","sob","sobbing","tragic","loss","lost"
    },
    "anger": {
        "angry","anger","mad","furious","irate","annoyed","pissed","rage","raging","fuming","hate","hateful",
        "despise","idiot","stupid","jerk","bastard","screw","kill","die","damn","dammit","annoy","annoying",
        "bad","rant","outrage","hostile","vengeful","attack","insult","smd","wtf"
    },
    "disgust": {
        "disgust","disgusting","disgusted","gross","nausea","nauseous","sick","vomit","barf","yuck","eww","icky",
        "filthy","rotten","sickening","repulsed","repulsive","ugh","bleh","cantstand","cant stand","grossedout",
        "ickyish","yucky","stinky","stink","putrid","odious"
    },
    "fear": {
        "fear","afraid","scared","scary","terrified","panic","anxious","anxiety","worried","dread","phobia",
        "nervous","panicattack","fright","frightened","terrify","paranoid","tremble","trembling","panicstruck"
    },
    "surprise": {
        "surprise","surprised","shocked","astonished","amazed","startled","wow","whoa","unbelievable","unexpected",
        "stunned","gasp","astounded","flabbergasted","jolt","jawdrop","ohmy","omg","no way","cantbelieve"
    }
}

EMO_EMOJI = {
    "joy": {"😀","😃","😄","😁","😊","☺️","😆","😂","🥳","🤩","😍","😘","🥰"},
    "sadness": {"☹️","🙁","😞","😔","😢","😭"},
    "anger": {"😠","😡","🤬","💢"},
    "disgust": {"🤢","🤮","🤧","😒"},
    "fear": {"😨","😰","😱","😧"},
    "surprise": {"😲","😮","😯","🤯"}
}

for emo, emjs in EMO_EMOJI.items():
    EMO_DICT.setdefault(emo, set()).update(emjs)

EMO_WORDS = {emo: set(words) for emo, words in EMO_DICT.items()}
ALL_WORDS = sorted(set().union(*EMO_WORDS.values()))

NEG_EXTRA = {
    "sadness": {"bored","boring","meh","bleh"},
    "anger": {"bad","annoy","annoying","jerk"},
    "disgust": {"disguisting","sucks","grossing"},
    "fear": {"scaredy","scarry","panicattack"}
}
for emo, words in NEG_EXTRA.items():
    EMO_WORDS.setdefault(emo, set()).update(words)
ALL_WORDS = sorted(set().union(*EMO_WORDS.values()))

# Dictionary matching
def find_emotion_from_dictionary(text: str, cutoff=0.72):
    if not isinstance(text, str):
        return None
    for emo, emjset in EMO_EMOJI.items():
        for emj in emjset:
            if emj in text:
                return emo

    cleaned = preprocess_text(text)
    tokens = re.findall(r"\w+", cleaned)
    if not tokens:
        return None

    for tok in tokens:
        for emo, words in EMO_WORDS.items():
            if tok in words:
                return emo

    joined = "".join(tokens)
    for emo, words in EMO_WORDS.items():
        if joined in words:
            return emo

    for tok in tokens:
        matches = difflib.get_close_matches(tok, ALL_WORDS, n=1, cutoff=cutoff)
        if matches:
            matched = matches[0]
            for emo, words in EMO_WORDS.items():
                if matched in words:
                    return emo

    if len(tokens) >= 2:
        for n in (2,3):
            for i in range(len(tokens)-n+1):
                gram = "".join(tokens[i:i+n])
                matches = difflib.get_close_matches(gram, ALL_WORDS, n=1, cutoff=cutoff)
                if matches:
                    matched = matches[0]
                    for emo, words in EMO_WORDS.items():
                        if matched in words:
                            return emo
    return None

# VADER fallback mapping
ANGER_KEYWORDS = {"hate","kill","die","idiot","stupid","jerk","bastard","screw","pissed","damn"}
DISGUST_KEYWORDS = {"disgust","gross","vomit","sick","yuck","eww","barf","filth","rotten"}

def vadermap_from_score(text: str, pos_thresh=0.55, neg_thresh=-0.55):
    vs = sia.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    if compound >= pos_thresh:
        return "joy"
    if compound <= neg_thresh:
        lowered = text.lower()
        tokens = set(re.findall(r"\w+", lowered))
        if tokens & ANGER_KEYWORDS:
            return "anger"
        if tokens & DISGUST_KEYWORDS:
            return "disgust"
        return "sadness"
    return None

# Load model artifacts
VEC_F = "vectorizer_emotion.pkl"
MODEL_F = "model_emotion.pkl"
LE_F = "label_encoder_emotion.pkl"

vec = None
model = None
le = None

if os.path.exists(VEC_F) and os.path.exists(MODEL_F) and os.path.exists(LE_F):
    try:
        vec = joblib.load(VEC_F)
        model = joblib.load(MODEL_F)
        le = joblib.load(LE_F)
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

# Combined prediction
def predict_text_combined(text: str):
    dict_label = find_emotion_from_dictionary(text, cutoff=0.72)
    if dict_label is not None:
        return dict_label, None

    vader_label = vadermap_from_score(text, pos_thresh=0.55, neg_thresh=-0.55)
    if vader_label is not None:
        return vader_label, None

    if model is not None and vec is not None and le is not None:
        clean = preprocess_text(text)
        X = vec.transform([clean])
        pidx = model.predict(X)[0]
        label = le.inverse_transform([pidx])[0]
        probs = None
        if hasattr(model, "predict_proba"):
            pr = model.predict_proba(X)[0]
            probs = dict(zip(le.inverse_transform(np.arange(len(pr))), pr.round(4)))
        return label, probs
    
    return "neutral", None

# Flask app
app = Flask(__name__)

EMOTIONS = {
    "joy": {"emoji": "😊", "color": "#FFD700", "hex": "#FFD700"},
    "sadness": {"emoji": "😢", "color": "#4A90E2", "hex": "#4A90E2"},
    "anger": {"emoji": "😠", "color": "#E74C3C", "hex": "#E74C3C"},
    "disgust": {"emoji": "🤮", "color": "#9B59B6", "hex": "#9B59B6"},
    "fear": {"emoji": "😨", "color": "#2C3E50", "hex": "#2C3E50"},
    "surprise": {"emoji": "😲", "color": "#F39C12", "hex": "#F39C12"},
    "neutral": {"emoji": "😐", "color": "#95A5A6", "hex": "#95A5A6"}
}

@app.route('/')
def index():
    return render_template('index.html', emotions=EMOTIONS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Please enter some text'}), 400
    
    emotion, probs = predict_text_combined(text)
    
    result = {
        'text': text,
        'emotion': emotion,
        'emoji': EMOTIONS.get(emotion, {}).get('emoji', '😐'),
        'color': EMOTIONS.get(emotion, {}).get('color', '#95A5A6'),
        'probabilities': probs
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

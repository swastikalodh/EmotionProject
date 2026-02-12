# app.py
"""
Flask web app for emotion detection.
- Loads saved artifacts if present (vectorizer_emotion.pkl, model_emotion.pkl, label_encoder_emotion.pkl).
- If missing, trains a model automatically (augmentation + LogisticRegression).
- Endpoint: GET / -> HTML UI
            POST /api/predict -> JSON prediction
Run:
    pip install flask joblib scikit-learn pandas nltk
    python app.py
"""

import os
import re
import random
import joblib
import difflib
import warnings
from functools import lru_cache

from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# NLTK (ensure resources)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required nltk data
for pkg in ("stopwords", "wordnet", "vader_lexicon"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# VADER
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

# Prepare preprocess helpers
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

KEEP = {"not", "no", "so", "very", "really", "too"}
STOP_WORDS = STOP_WORDS.difference(KEEP)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str):
    if text is None:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s'😀-🙏]", " ", t)
    tokens = t.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOP_WORDS]
    return " ".join(tokens)

# Build a compact emotion dictionary (same core as your system; you can expand)
EMO_DICT = {
    "joy": {"happy","joy","joyful","glad","delight","delighted","cheerful","smile","smiling","excited","yay","love"},
    "sadness": {"sad","sadness","sorrow","lonely","cry","crying","tears","depressed","bored","boring","meh"},
    "anger": {"angry","anger","mad","furious","hate","idiot","stupid","jerk","bad","annoy","annoying"},
    "disgust": {"disgust","disgusting","gross","sick","vomit","eww","yuck","barf","filth"},
    "fear": {"fear","afraid","scared","panic","anxious","worried","terrified"},
    "surprise": {"surprise","surprised","shocked","amazed","wow","whoa","unexpected"}
}
EMO_EMOJI = {
    "joy": {"😀","😃","😄","😁","😊","😂","🥳","😍","😘","🥰"},
    "sadness": {"☹️","🙁","😞","😔","😢","😭"},
    "anger": {"😠","😡","🤬"},
    "disgust": {"🤢","🤮"},
    "fear": {"😨","😰","😱"},
    "surprise": {"😲","😮","🤯"}
}
for emo, emjs in EMO_EMOJI.items():
    EMO_DICT.setdefault(emo, set()).update(emjs)

EMO_WORDS = {emo: set(words) for emo, words in EMO_DICT.items()}
ALL_WORDS = sorted(set().union(*EMO_WORDS.values()))

def find_emotion_from_dictionary(text: str, cutoff=0.72):
    if not isinstance(text, str):
        return None
    # emojis
    for emo, emjs in EMO_EMOJI.items():
        for emj in emjs:
            if emj in text:
                return emo
    cleaned = preprocess_text(text)
    tokens = re.findall(r"\w+", cleaned)
    if not tokens:
        return None
    # exact
    for tok in tokens:
        for emo, words in EMO_WORDS.items():
            if tok in words:
                return emo
    # joined
    joined = "".join(tokens)
    for emo, words in EMO_WORDS.items():
        if joined in words:
            return emo
    # close-match
    for tok in tokens:
        matches = difflib.get_close_matches(tok, ALL_WORDS, n=1, cutoff=cutoff)
        if matches:
            matched = matches[0]
            for emo, words in EMO_WORDS.items():
                if matched in words:
                    return emo
    return None

# VADER fallback mapping
ANGER_KEYWORDS = {"hate","kill","die","idiot","stupid","jerk","bastard","screw","pissed"}
DISGUST_KEYWORDS = {"disgust","gross","vomit","sick","yuck","eww","barf","filth"}

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

# Artifact filenames
VEC_F = "vectorizer_emotion.pkl"
MODEL_F = "model_emotion.pkl"
LE_F = "label_encoder_emotion.pkl"

# Train function (same augmentation approach)
def train_and_save_artifacts(force_retrain=False):
    if not force_retrain and os.path.exists(VEC_F) and os.path.exists(MODEL_F) and os.path.exists(LE_F):
        return  # nothing to do
    # Create minimal csv if missing
    csv = "emotions.csv"
    if not os.path.exists(csv):
        samples = [
            ("I feel so happy today!","joy"),("This is the best day ever!","joy"),("I am excited and joyful.","joy"),
            ("I am really sad.","sadness"),("Everything is falling apart.","sadness"),
            ("I am so angry!","anger"),("This terrifies me","fear"),
            ("That's disgusting","disgust"),("What a surprise!","surprise")
        ]
        pd.DataFrame(samples, columns=["text","label"]).to_csv(csv, index=False)
    df = pd.read_csv(csv)

    # augmentation (compact version)
    joy_templates = ["I am {} today","I feel {}","This makes me feel {}","So {}","I am very {}","Feeling {}"]
    joy_words = list(EMO_WORDS["joy"]) * 5
    anger_templates = ["I hate {}","{} pisses me off","I am furious at {}","I despise {}","I can't stand {}","I hate you"]
    anger_targets = ["him","her","them","you","this","that","it","someone"]
    disgust_templates = ["That is disgusting","I feel disgusted by {}","This is gross","This smells awful","So gross {}"]
    sadness_templates = ["I feel so sad","I am heartbroken","I just want to cry","I feel depressed","This makes me miserable"]
    fear_templates = ["I am so scared","I feel terrified","I am afraid of {}","This frightens me","I panic about {}"]
    surprise_templates = ["I did not expect that","Wow that's surprising","What a shock","I am astonished","That shocked me"]

    aug = []
    N_JOY = 300
    N_NEG = 200

    for _ in range(N_JOY):
        t = random.choice(joy_templates)
        w = random.choice(joy_words)
        if random.random() < 0.3:
            w = random.choice(["so "+w, "very "+w, "really "+w])
        aug.append((t.format(w), "joy"))
    for _ in range(N_NEG):
        t = random.choice(anger_templates)
        if "{}" in t:
            aug.append((t.format(random.choice(anger_targets)), "anger"))
        else:
            aug.append((t, "anger"))
    for _ in range(N_NEG):
        t = random.choice(disgust_templates)
        if "{}" in t:
            aug.append((t.format(random.choice(["that","this","the food","the smell"])), "disgust"))
        else:
            aug.append((t, "disgust"))
    for _ in range(N_NEG):
        aug.append((random.choice(sadness_templates), "sadness"))
    for _ in range(N_NEG):
        aug.append((random.choice(fear_templates).format(random.choice(["this","that","the event"])), "fear"))
    for _ in range(N_NEG):
        aug.append((random.choice(surprise_templates), "surprise"))

    aug_df = pd.DataFrame(aug, columns=["text","label"])
    df_full = pd.concat([df, aug_df]).reset_index(drop=True)
    df_full["clean"] = df_full["text"].apply(preprocess_text)

    # balance
    max_count = df_full["label"].value_counts().max()
    frames = []
    for lbl, grp in df_full.groupby("label"):
        if len(grp) < max_count:
            frames.append(resample(grp, replace=True, n_samples=max_count, random_state=42))
        else:
            frames.append(grp)
    df_bal = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

    # vectorize + train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = TfidfVectorizer(max_features=12000, ngram_range=(1,2))
        X = vec.fit_transform(df_bal["clean"])
        le = LabelEncoder()
        y = le.fit_transform(df_bal["label"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        model = LogisticRegression(max_iter=4000, class_weight="balanced")
        model.fit(X_train, y_train)
    # save
    joblib.dump(vec, VEC_F)
    joblib.dump(model, MODEL_F)
    joblib.dump(le, LE_F)

# Ensure artifacts exist at startup (non-blocking training if missing)
if not (os.path.exists(VEC_F) and os.path.exists(MODEL_F) and os.path.exists(LE_F)):
    print("[INFO] Artifacts missing — training model now (this may take a minute)...")
    train_and_save_artifacts()
else:
    print("[INFO] Found existing artifacts; Flask will load them.")

# Load artifacts into memory
vec = joblib.load(VEC_F)
model = joblib.load(MODEL_F)
le = joblib.load(LE_F)

# prediction flow: dictionary -> VADER -> model
def predict_text_combined(text: str):
    dict_label = find_emotion_from_dictionary(text, cutoff=0.72)
    if dict_label is not None:
        return dict_label, None
    vader_label = vadermap_from_score(text, pos_thresh=0.55, neg_thresh=-0.55)
    if vader_label is not None:
        return vader_label, None
    clean = preprocess_text(text)
    X = vec.transform([clean])
    pidx = model.predict(X)[0]
    label = le.inverse_transform([pidx])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        pr = model.predict_proba(X)[0]
        probs = dict(zip(le.inverse_transform(np.arange(len(pr))), pr.round(4)))
    return label, probs

# Flask app
app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or request.form
    text = data.get("text") if isinstance(data, dict) else request.form.get("text", "")
    if not text:
        return jsonify({"error":"no text provided"}), 400
    label, probs = predict_text_combined(text)
    resp = {"predicted": label}
    if probs is not None:
        resp["probabilities"] = probs
    return jsonify(resp)

if __name__ == "__main__":
    # run flask
    app.run(debug=True)

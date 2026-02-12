# emotion_detection.py
# Standalone emotion detection script with robust stratified train/test sizing.
# Run: python emotion_detection.py

import os
import pandas as pd
import string
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
try:
    _ = stopwords.words("english")
except:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not", "no", "so", "very", "really", "too", "don't", "can't", "won't"}
STOP_WORDS = STOP_WORDS.difference(KEEP)

lemmatizer = WordNetLemmatizer()

# ---------- preprocessing ----------
def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    clean = []
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch == "'":
            clean.append(ch)
        else:
            clean.append(" ")
    text = "".join(clean)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

# ---------- create simple dataset ----------
def create_csv():
    if os.path.exists("emotions.csv"):
        print("[INFO] emotions.csv already exists.")
        return
    data = {
        "text": [
            # JOY
            "I feel so happy today!",
            "This is the best day ever!",
            "I am excited and joyful.",
            "I can't stop smiling.",
            "Everything feels wonderful!",

            # SADNESS
            "I feel so sad and lonely.",
            "Everything is falling apart.",
            "I just want to cry.",
            "I feel empty inside.",
            "Nothing seems right today.",

            # ANGER
            "I am so angry right now!",
            "This makes me furious!",
            "I can't control my anger.",
            "I am pissed off!",
            "Stop bothering me!",

            # FEAR
            "I am scared and worried.",
            "This situation terrifies me.",
            "I feel afraid of what might happen.",
            "My heart is racing from fear.",
            "I am really frightened.",

            # DISGUST
            "That is so disgusting!",
            "I feel grossed out.",
            "This smells awful!",
            "I can't even look at it.",
            "That food tasted horrible.",

            # SURPRISE
            "Wow, I did not expect that!",
            "I am shocked right now!",
            "That was surprising!",
            "I can't believe that happened!",
            "This is unbelievable!"
        ],
        "label": [
            "joy","joy","joy","joy","joy",
            "sadness","sadness","sadness","sadness","sadness",
            "anger","anger","anger","anger","anger",
            "fear","fear","fear","fear","fear",
            "disgust","disgust","disgust","disgust","disgust",
            "surprise","surprise","surprise","surprise","surprise"
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv("emotions.csv", index=False)
    print("[INFO] Created emotions.csv with simple dataset.")

# ---------- balance ----------
def balance(df):
    max_size = df["label"].value_counts().max()
    frames = []
    for emotion in df["label"].unique():
        subset = df[df["label"] == emotion]
        if len(subset) < max_size:
            upsampled = resample(subset, replace=True, n_samples=max_size, random_state=42)
            frames.append(upsampled)
        else:
            frames.append(subset)
    balanced_df = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[INFO] Balanced dataset to {max_size} per emotion (total {len(balanced_df)} samples).")
    return balanced_df

# ---------- compute safe test size ----------
def compute_safe_test_size(n_samples, n_classes, fraction=0.20):
    """Return an integer test_size suitable for stratified split:
       ensure test_size_int >= n_classes and >= round(fraction * n_samples).
    """
    # compute desired by fraction (rounded)
    desired_by_frac = max(1, int(round(fraction * n_samples)))
    test_size_int = max(n_classes, desired_by_frac)
    # but test size must be less than n_samples
    if test_size_int >= n_samples:
        # leave at least one sample for training per class if possible
        test_size_int = max(n_classes, n_samples // 3)
        if test_size_int >= n_samples:
            # fallback: use 1 for very tiny datasets
            test_size_int = max(1, n_samples - n_classes)
            if test_size_int < 1:
                test_size_int = 1
    return test_size_int

# ---------- training ----------
def train_model():
    create_csv()
    df = pd.read_csv("emotions.csv")
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("emotions.csv must contain 'text' and 'label' columns.")
    df["clean"] = df["text"].apply(preprocess)
    df_bal = balance(df)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_bal["clean"])

    le = LabelEncoder()
    y = le.fit_transform(df_bal["label"])

    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    test_size_int = compute_safe_test_size(n_samples, n_classes, fraction=0.20)

    # If test_size_int is an integer, call train_test_split with test_size=test_size_int
    # else (shouldn't happen) fallback to fraction
    print(f"[INFO] n_samples={n_samples}, n_classes={n_classes}, test_size_int={test_size_int}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_int, random_state=42, stratify=y
    )

    print(f"[INFO] Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = LogisticRegression(max_iter=4000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))

    # save
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("[INFO] Saved model.pkl, vectorizer.pkl, label_encoder.pkl")

    return model, vectorizer, le

# ---------- test ----------
def test_prediction(model, vectorizer, le):
    test_text = "I feel so happy today!"
    clean = preprocess(test_text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]

    print("\n=== TEST SENTENCE ===")
    print("Input:", test_text)
    print("Predicted Emotion:", label)
    # show probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_map = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
        print("Probabilities:", prob_map)

if __name__ == "__main__":
    print("[INFO] Training model...")
    model, vectorizer, le = train_model()
    print("\n[INFO] Testing model with sample text...")
    test_prediction(model, vectorizer, le)
    print("\n[INFO] Done.")

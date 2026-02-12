# emotion_detection.py
# Emotion detection + interactive input in VS Code terminal.

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

# Setup NLTK
try:
    _ = stopwords.words("english")
except:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not", "no", "so", "very", "really", "too"}
STOP_WORDS = STOP_WORDS.difference(KEEP)
lemmatizer = WordNetLemmatizer()

# ---------------------------
# PREPROCESSING
# ---------------------------
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

# ---------------------------
# CSV CREATION
# ---------------------------
def create_csv():
    if os.path.exists("emotions.csv"):
        return

    data = {
        "text": [
            "I feel so happy today!",
            "This is the best day ever!",
            "I can't stop smiling.",
            "Everything feels wonderful!",
            "I am excited and joyful.",

            "I feel so sad and lonely.",
            "Everything is falling apart.",
            "I just want to cry.",
            "I feel empty inside.",
            "Nothing seems right today.",

            "I am so angry right now!",
            "This makes me furious!",
            "I can't control my anger.",
            "I am pissed off!",
            "Stop bothering me!",

            "I am scared and worried.",
            "This situation terrifies me.",
            "I feel afraid of what might happen.",
            "My heart is racing from fear.",
            "I am really frightened.",

            "That is so disgusting!",
            "This smells awful!",
            "I feel grossed out.",
            "I can't even look at it.",
            "That food tasted horrible.",

            "Wow, I did not expect that!",
            "That was surprising!",
            "I am shocked right now!",
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
    print("[INFO] emotions.csv created.")

# ---------------------------
# BALANCE DATA
# ---------------------------
def balance(df):
    max_size = df["label"].value_counts().max()
    frames = []
    for lbl in df["label"].unique():
        grp = df[df["label"] == lbl]
        if len(grp) < max_size:
            frames.append(resample(grp, replace=True, n_samples=max_size, random_state=42))
        else:
            frames.append(grp)
    return pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------
# TRAIN MODEL
# ---------------------------
def train_model():
    create_csv()

    df = pd.read_csv("emotions.csv")
    df["clean"] = df["text"].apply(preprocess)

    df = balance(df)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(df["clean"])

    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=6, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=4000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))

    joblib.dump(model, "model.pkl")
    joblib.dump(vec, "vectorizer.pkl")
    joblib.dump(le, "label_encoder.pkl")

    print("[INFO] Model saved.")

    return model, vec, le

# ---------------------------
# INTERACTIVE PREDICTION
# ---------------------------
def interactive_mode(model, vec, le):
    print("\n==============================")
    print(" Emotion Detection - Interactive Mode")
    print("==============================")
    print("Type a sentence and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        clean = preprocess(user_input)
        X = vec.transform([clean])
        pred = model.predict(X)[0]
        label = le.inverse_transform([pred])[0]

        print("Predicted Emotion:", label, "\n")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    model, vec, le = train_model()
    interactive_mode(model, vec, le)

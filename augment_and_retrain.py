
# augment_and_retrain_fixed.py
"""
Augment, retrain, save artifacts, and allow interactive prediction from terminal.
Includes a rule-based fallback to correctly detect strong negative phrases (e.g. "i hate you").
Run: python augment_and_retrain_fixed.py
"""

import os
import random
import joblib
import pandas as pd
import numpy as np
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# NLP helpers
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
try:
    _ = stopwords.words("english")
except Exception:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not", "no", "so", "very", "really"}
STOP_WORDS = STOP_WORDS.difference(KEEP)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str):
    """Lowercase, keep apostrophes, remove other punctuation, lemmatize, remove stopwords (but keep negations)."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # keep apostrophes for contractions; replace other punctuation with spaces
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

# Path & CSV
CSV = "emotions.csv"
def ensure_csv():
    if not os.path.exists(CSV):
        print("[INFO] No emotions.csv found — creating a small default dataset.")
        samples = [
            ("I feel so happy today!","joy"),("This is the best day ever!","joy"),("I am excited and joyful.","joy"),
            ("I am really sad.","sadness"),("Everything is falling apart.","sadness"),
            ("I am so angry!", "anger"),("This terrifies me","fear"),
            ("That's disgusting","disgust"),("What a surprise!", "surprise")
        ]
        df = pd.DataFrame(samples, columns=["text","label"])
        df.to_csv(CSV, index=False)
    else:
        print(f"[INFO] Found existing {CSV}")

ensure_csv()
df = pd.read_csv(CSV)

# ----------------------------
# AUGMENT: joy + negative templates
# ----------------------------
joy_templates = [
    "I am {} today", "I feel {}", "This makes me feel {}", "I'm so {} right now", "{} and thrilled",
    "Absolutely {}", "I'm feeling really {}", "So {}", "Can't stop feeling {}"
]
joy_synonyms = ["happy","joyful","delighted","elated","ecstatic","thrilled","overjoyed","cheerful","glad","content"]

# Negative templates for anger/disgust/sadness
anger_templates = [
    "I hate you", "I hate {}","I want to hurt {}","{} pisses me off","You are so {}",
    "I am furious at {}", "I will kill {}", "I despise {}", "I can't stand {}","Go to hell {}"
]
anger_targets = ["him","her","them","you","this","that","it","himself","herself","someone"]

disgust_templates = [
    "That is disgusting", "I feel disgusted by {}", "This is gross", "This smells awful",
    "I feel sick looking at {}", "So gross {}"
]
sadness_templates = [
    "I feel so sad", "I am heartbroken", "I just want to cry", "I am depressed", "This makes me miserable"
]

# generate synthetic examples
N_JOY = 300
N_NEG_PER_CLASS = 200  # number of synthetic negative-type examples per negative class

aug_texts = []
# Joy augmentation
for _ in range(N_JOY):
    t = random.choice(joy_templates)
    s = random.choice(joy_synonyms)
    if random.random() < 0.35:
        s = random.choice(["so "+s, "very "+s, "really "+s])
    aug_texts.append((t.format(s), "joy"))

# Anger augmentation (varied)
for _ in range(N_NEG_PER_CLASS):
    t = random.choice(anger_templates)
    # sometimes format a target, sometimes use the phrase as-is
    if "{}" in t:
        target = random.choice(anger_targets)
        aug_texts.append((t.format(target), "anger"))
    else:
        aug_texts.append((t, "anger"))

# Disgust augmentation
for _ in range(N_NEG_PER_CLASS):
    t = random.choice(disgust_templates)
    if "{}" in t:
        # choose some object/target
        obj = random.choice(["that", "this", "food", "the sight", "the smell", "the taste"])
        aug_texts.append((t.format(obj), "disgust"))
    else:
        aug_texts.append((t, "disgust"))

# Sadness augmentation
for _ in range(N_NEG_PER_CLASS):
    t = random.choice(sadness_templates)
    aug_texts.append((t, "sadness"))

print(f"[INFO] Generated synthetic examples: joy={N_JOY}, anger/disgust/sadness={N_NEG_PER_CLASS} each")
aug_df = pd.DataFrame(aug_texts, columns=["text","label"])

# Merge and preprocess
df_full = pd.concat([df, aug_df]).reset_index(drop=True)
df_full["clean"] = df_full["text"].apply(preprocess_text)

# Balance classes by upsampling
max_count = df_full["label"].value_counts().max()
frames = []
for lbl, grp in df_full.groupby("label"):
    if len(grp) < max_count:
        frames.append(resample(grp, replace=True, n_samples=max_count, random_state=42))
    else:
        frames.append(grp)
df_bal = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
print("[INFO] After augmentation and balancing, class counts:")
print(df_bal["label"].value_counts())

# ----------------------------
# Train or load artifacts
# ----------------------------
VEC_FILE = "vectorizer_fixed.pkl"
MODEL_FILE = "model_fixed.pkl"
LE_FILE = "label_encoder_fixed.pkl"

retrain = True
# if artifacts exist, prompt user
if os.path.exists(MODEL_FILE) and os.path.exists(VEC_FILE) and os.path.exists(LE_FILE):
    print("[INFO] Found existing trained artifacts.")
    ch = input("Type 'r' to retrain, Enter to reuse existing model: ").strip().lower()
    if ch != 'r':
        retrain = False

if retrain:
    print("[INFO] Training new model...")
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X = vec.fit_transform(df_bal["clean"])
    le = LabelEncoder()
    y = le.fit_transform(df_bal["label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    print("\n[RESULT] Accuracy after augmentation:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save new artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vec, VEC_FILE)
    joblib.dump(le, LE_FILE)
    print("[INFO] Saved", MODEL_FILE, VEC_FILE, LE_FILE)
else:
    print("[INFO] Loading existing artifacts...")
    model = joblib.load(MODEL_FILE)
    vec = joblib.load(VEC_FILE)
    le = joblib.load(LE_FILE)

# ----------------------------
# RULE-BASED FALLBACK
# ----------------------------
# Strong negative keywords mapping (simple)
ANGER_KEYWORDS = {"hate", "kill", "murder", "die", "die!", "screw", "scum", "idiot", "stupid", "bastard", "trash", "annoy"}
DISGUST_KEYWORDS = {"disgust", "gross", "nausea", "vomit", "ick", "eww", "smell", "rotten"}
SADNESS_KEYWORDS = {"sad", "depress", "lonely", "cry", "suicid", "hopeless", "miserable"}

def rule_based_label(text):
    """Return label if strong lexical signals exist, otherwise None."""
    t = text.lower()
    # direct matches for classic short insults (fast path)
    if re.search(r"\bi hate (you|him|her|them|it)\b", t):
        return "anger"
    # contains 'i hate' anywhere
    if re.search(r"\bi hate\b", t):
        return "anger"
    # presence of strong keywords
    tokens = set(re.findall(r"\w+", t))
    if tokens & ANGER_KEYWORDS:
        return "anger"
    if tokens & DISGUST_KEYWORDS:
        return "disgust"
    if tokens & SADNESS_KEYWORDS:
        return "sadness"
    return None

# ----------------------------
# Quick test (summary)
# ----------------------------
test = "I feel so happy today!"
test_clean = preprocess_text(test)
X_t = vec.transform([test_clean])
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_t)[0]
    prob_map = dict(zip(le.inverse_transform(np.arange(len(probs))), probs.round(4)))
else:
    prob_map = None
pred_label = le.inverse_transform([model.predict(X_t)[0]])[0]
print("\n[INFO] Quick test")
print("Test:", test)
if prob_map is not None:
    print("Probabilities:", prob_map)
print("Predicted:", pred_label)

# ----------------------------
# Interactive loop (with rule fallback)
# ----------------------------
print("\n=== Interactive mode ===")
print('Type a sentence and press Enter. Type "exit" to quit.')

def inspect_and_predict(user_text):
    # rule-based check first
    rule = rule_based_label(user_text)
    if rule is not None:
        return {"predicted": rule, "method": "RULE", "probs": None}

    # otherwise use model
    clean = preprocess_text(user_text)
    X_u = vec.transform([clean])
    pred_idx = model.predict(X_u)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    out = {"predicted": pred_label, "method": "MODEL"}
    if hasattr(model, "predict_proba"):
        probs_u = model.predict_proba(X_u)[0]
        out["probs"] = dict(zip(le.inverse_transform(np.arange(len(probs_u))), probs_u.round(4)))
    return out

try:
    while True:
        user_input = input("\nEnter text (or 'exit' to quit): ").strip()
        if user_input.lower() in ("exit","quit"):
            print("Goodbye!")
            break
        if user_input == "":
            print("Please type something or 'exit' to quit.")
            continue
        result = inspect_and_predict(user_input)
        print("Predicted Emotion:", result["predicted"], f"(by {result['method']})")
        if result.get("probs") is not None:
            print("Probabilities:", result["probs"])
except KeyboardInterrupt:
    print("\nInterrupted. Bye.")


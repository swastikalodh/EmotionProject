# emotion_system.py
"""
Comprehensive emotion detection with:
- Large emotion dictionary + emoji support (typo tolerant)
- VADER sentiment fallback (auto-downloads vader_lexicon if needed)
- Augment + train TF-IDF + LogisticRegression model
- Interactive terminal
Run: python emotion_system.py
"""

import os
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

# NLTK helpers
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# attempt to ensure needed NLTK resources (quiet)
for pkg in ("stopwords", "wordnet"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# VADER: ensure vader_lexicon is present then instantiate SentimentIntensityAnalyzer
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        # try instantiate (may raise LookupError if vader missing)
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
except Exception:
    # If something goes wrong, define a fallback SIA that always returns neutral
    class _DummySIA:
        def polarity_scores(self, text):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    sia = _DummySIA()

# Prepare stopwords/lemmatizer
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

KEEP = {"not", "no", "so", "very", "really", "too"}
STOP_WORDS = STOP_WORDS.difference(KEEP)
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    # keep emojis and apostrophes; remove other punctuation
    t = re.sub(r"[^\w\s'😀-🙏]", " ", t)
    tokens = t.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOP_WORDS]
    return " ".join(tokens)

# ----------------------------
# Large emotion dictionary + emoji mapping
# ----------------------------
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

# Emojis mapping
EMO_EMOJI = {
    "joy": {"😀","😃","😄","😁","😊","☺️","😆","😂","🥳","🤩","😍","😘","🥰"},
    "sadness": {"☹️","🙁","😞","😔","😢","😭"},
    "anger": {"😠","😡","🤬","💢"},
    "disgust": {"🤢","🤮","🤧","😒"},
    "fear": {"😨","😰","😱","😧"},
    "surprise": {"😲","😮","😯","🤯"}
}

# Merge emoji into EMO_DICT
for emo, emjs in EMO_EMOJI.items():
    EMO_DICT.setdefault(emo, set()).update(emjs)

# Normalize
EMO_WORDS = {emo: set(words) for emo, words in EMO_DICT.items()}
ALL_WORDS = sorted(set().union(*EMO_WORDS.values()))

# Add a few extra negative tokens explicitly (helps short phrases)
NEG_EXTRA = {
    "sadness": {"bored","boring","meh","bleh"},
    "anger": {"bad","annoy","annoying","jerk"},
    "disgust": {"disguisting","sucks","grossing"},
    "fear": {"scaredy","scarry","panicattack"}
}
for emo, words in NEG_EXTRA.items():
    EMO_WORDS.setdefault(emo, set()).update(words)
ALL_WORDS = sorted(set().union(*EMO_WORDS.values()))

# ----------------------------
# Dictionary matching (typo tolerant)
# ----------------------------
def find_emotion_from_dictionary(text: str, cutoff=0.72):
    if not isinstance(text, str):
        return None
    # emoji quick detection in raw text
    for emo, emjset in EMO_EMOJI.items():
        for emj in emjset:
            if emj in text:
                return emo

    cleaned = preprocess_text(text)
    tokens = re.findall(r"\w+", cleaned)
    if not tokens:
        return None

    # exact token match
    for tok in tokens:
        for emo, words in EMO_WORDS.items():
            if tok in words:
                return emo

    # joined token match
    joined = "".join(tokens)
    for emo, words in EMO_WORDS.items():
        if joined in words:
            return emo

    # close match on tokens
    for tok in tokens:
        matches = difflib.get_close_matches(tok, ALL_WORDS, n=1, cutoff=cutoff)
        if matches:
            matched = matches[0]
            for emo, words in EMO_WORDS.items():
                if matched in words:
                    return emo

    # bigram/trigram combined
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

# ----------------------------
# VADER fallback mapping
# ----------------------------
ANGER_KEYWORDS = {"hate","kill","die","idiot","stupid","jerk","bastard","screw","pissed","damn"}
DISGUST_KEYWORDS = {"disgust","gross","vomit","sick","yuck","eww","barf","filth","rotten"}

def vadermap_from_score(text: str, pos_thresh=0.55, neg_thresh=-0.55):
    vs = sia.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    # strong positive -> joy
    if compound >= pos_thresh:
        return "joy"
    # strong negative -> try to map to anger or disgust or sadness
    if compound <= neg_thresh:
        lowered = text.lower()
        tokens = set(re.findall(r"\w+", lowered))
        if tokens & ANGER_KEYWORDS:
            return "anger"
        if tokens & DISGUST_KEYWORDS:
            return "disgust"
        return "sadness"
    return None

# ----------------------------
# Dataset, augmentation, training (LogisticRegression)
# ----------------------------
CSV = "emotions.csv"
def ensure_csv_minimal():
    if not os.path.exists(CSV):
        print("[INFO] Creating minimal emotions.csv.")
        samples = [
            ("I feel so happy today!","joy"),("This is the best day ever!","joy"),("I am excited and joyful.","joy"),
            ("I am really sad.","sadness"),("Everything is falling apart.","sadness"),
            ("I am so angry!","anger"),("This terrifies me","fear"),
            ("That's disgusting","disgust"),("What a surprise!","surprise")
        ]
        df = pd.DataFrame(samples, columns=["text","label"])
        df.to_csv(CSV, index=False)
    else:
        print(f"[INFO] Found {CSV}")

ensure_csv_minimal()
df = pd.read_csv(CSV)

# augmentation templates
joy_templates = ["I am {} today","I feel {}","This makes me feel {}","So {}","I am very {}","Feeling {}"]
joy_words = list(EMO_WORDS["joy"])[:80]

anger_templates = ["I hate {}","{} pisses me off","I am furious at {}","I despise {}","I can't stand {}","I hate you"]
anger_targets = ["him","her","them","you","this","that","it","someone","the person"]
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

# balance by upsampling
max_count = df_full["label"].value_counts().max()
frames = []
for lbl, grp in df_full.groupby("label"):
    if len(grp) < max_count:
        frames.append(resample(grp, replace=True, n_samples=max_count, random_state=42))
    else:
        frames.append(grp)
df_bal = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
print("[INFO] After augmentation & balancing, class counts:")
print(df_bal["label"].value_counts())

# artifacts
VEC_F = "vectorizer_emotion.pkl"
MODEL_F = "model_emotion.pkl"
LE_F = "label_encoder_emotion.pkl"

retrain = True
if os.path.exists(VEC_F) and os.path.exists(MODEL_F) and os.path.exists(LE_F):
    print("[INFO] Found existing artifacts.")
    ch = input("Press Enter to reuse model, or type 'r' to retrain: ").strip().lower()
    if ch != "r":
        retrain = False

if retrain:
    print("[INFO] Training LogisticRegression model (this may take a moment)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = TfidfVectorizer(max_features=12000, ngram_range=(1,2))
        X = vec.fit_transform(df_bal["clean"])
        le = LabelEncoder()
        y = le.fit_transform(df_bal["label"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        model = LogisticRegression(max_iter=4000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    print("\n[RESULT] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    joblib.dump(vec, VEC_F)
    joblib.dump(model, MODEL_F)
    joblib.dump(le, LE_F)
    print("[INFO] Saved artifacts.")
else:
    print("[INFO] Loading artifacts...")
    vec = joblib.load(VEC_F)
    model = joblib.load(MODEL_F)
    le = joblib.load(LE_F)

# quick test
def quick_test():
    s = "I feel so happy today!"
    clean = preprocess_text(s)
    Xs = vec.transform([clean])
    pred = le.inverse_transform([model.predict(Xs)[0]])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        pr = model.predict_proba(Xs)[0]
        probs = dict(zip(le.inverse_transform(np.arange(len(pr))), pr.round(4)))
    print("\n[INFO] Quick test:", s)
    if probs:
        print("Probabilities:", probs)
    print("Predicted:", pred)

quick_test()

# ----------------------------
# Combined prediction flow: dictionary -> VADER -> model
# ----------------------------
def predict_text_combined(text: str):
    # 1) dictionary
    dict_label = find_emotion_from_dictionary(text, cutoff=0.72)
    if dict_label is not None:
        return dict_label, None

    # 2) VADER sentiment
    vader_label = vadermap_from_score(text, pos_thresh=0.55, neg_thresh=-0.55)
    if vader_label is not None:
        return vader_label, None

    # 3) ML model
    clean = preprocess_text(text)
    X = vec.transform([clean])
    pidx = model.predict(X)[0]
    label = le.inverse_transform([pidx])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        pr = model.predict_proba(X)[0]
        probs = dict(zip(le.inverse_transform(np.arange(len(pr))), pr.round(4)))
    return label, probs

# ----------------------------
# Interactive loop
# ----------------------------
print("\n=== Interactive Mode ===")
print("Type sentences and press Enter. Type 'exit' to quit.")
try:
    while True:
        txt = input("\nEnter text (or 'exit'): ").strip()
        if txt.lower() in ("exit","quit"):
            print("Goodbye!")
            break
        if txt == "":
            print("Please type something or 'exit'.")
            continue
        label, probs = predict_text_combined(txt)
        print("Predicted Emotion:", label)
        if probs is not None:
            print("Probabilities:", probs)
except KeyboardInterrupt:
    print("\nInterrupted. Bye.")



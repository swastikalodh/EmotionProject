"""
🌌 Vibe Oracle — Full Streamlit App
Detects emotional vibe of user input text with a mystical, visually rich UI.

Stack: streamlit==1.34.0 | nltk==3.8.1 | scikit-learn | pandas | numpy | joblib | deep-translator
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import os
import tempfile

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ── Download required NLTK data ───────────────────────────────────────────────
for _pkg in ["vader_lexicon", "stopwords", "wordnet", "punkt", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

# =============================================================================
# EMOTION DATA DICTIONARIES
# =============================================================================

EMOTION_KEYWORDS = {
    "joy": [
        "happy", "happiness", "joyful", "excited", "love", "wonderful", "amazing",
        "great", "fantastic", "cheerful", "delighted", "thrilled", "bliss", "elated",
        "ecstatic", "glad", "laugh", "smile", "celebrate", "fun", "enjoy", "grateful",
        "awesome", "brilliant", "content", "pleased", "radiant", "euphoric", "good",
        "excellent", "positive", "hopeful", "vibrant", "alive", "bright",
    ],
    "anger": [
        "angry", "anger", "furious", "rage", "mad", "hate", "irritated", "annoyed",
        "outraged", "frustrated", "enraged", "livid", "fuming", "hostile", "bitter",
        "resentful", "aggressive", "violent", "disgusted", "infuriated", "explode",
        "boiling", "seething", "wrathful", "irate", "temper", "snap", "explosive",
    ],
    "sadness": [
        "sad", "unhappy", "depressed", "miserable", "heartbroken", "grief", "cry",
        "sorrow", "mournful", "hopeless", "lonely", "gloomy", "melancholy", "despair",
        "desolate", "tragic", "painful", "lost", "tears", "devastated", "anguish",
        "down", "blue", "broken", "suffering", "hurt", "empty", "void", "miss",
    ],
    "fear": [
        "afraid", "fear", "scared", "terrified", "anxious", "nervous", "panic",
        "dread", "horror", "terror", "phobia", "worried", "uneasy", "apprehensive",
        "trembling", "fright", "nightmare", "shock", "startled", "petrified",
        "paranoid", "shaking", "tremble", "spooked", "creepy", "haunted",
    ],
    "disgust": [
        "disgusting", "gross", "revolting", "nasty", "awful", "yuck", "repulsed",
        "sick", "vomit", "nauseating", "horrible", "repulsive", "filthy", "foul",
        "unpleasant", "loathe", "abhorrent", "putrid", "hideous", "revolted",
        "sickening", "vile", "repugnant", "offensive", "stink",
    ],
    "surprise": [
        "surprised", "shocked", "astonished", "amazed", "unexpected", "wow",
        "unbelievable", "incredible", "stunning", "remarkable", "astounded",
        "speechless", "gasp", "omg", "whoa", "sudden", "startling", "jaw-dropping",
        "mind-blowing", "extraordinary", "unreal", "whoah", "no way",
    ],
}

# Hinglish + Bengali multi-language phrases (checked against raw input)
MULTILANG_PHRASES = {
    "joy": [
        "bahut maza", "kitna maza", "so happy", "bohot khushi", "maja aa gaya",
        "full masti", "dil khush", "ek number", "bhai wah", "acha lag raha",
        "\u0985\u09a8\u09c7\u0995 \u09ae\u099c\u09be",  # অনেক মজা
        "\u0996\u09c1\u09ac \u09ad\u09be\u09b2\u09cb",  # খুব ভালো
        "\u0986\u09a8\u09a8\u09cd\u09a6",               # আনন্দ
        "\u09b9\u09be\u09b8\u09bf",                     # হাসি
        "\u09a6\u09be\u09b0\u09c1\u09a3",               # দারুণ
    ],
    "anger": [
        "bahut gussa", "bura lag raha", "kuch nahi chahiye", "bohot bura",
        "chup raho", "teri toh", "faltu baat", "kya bakwas", "dimag mat kha",
        "\u09b0\u09be\u0997 \u09b9\u099a\u09cd\u099b\u09c7",  # রাগ হচ্ছে
        "\u0996\u09c1\u09ac \u09b0\u09be\u0997",              # খুব রাগ
        "\u09ac\u09bf\u09b0\u0995\u09cd\u09a4",               # বিরক্ত
        "\u0998\u09c7\u09a8\u09cd\u09a8\u09be",               # ঘেন্না
    ],
    "sadness": [
        "bahut dukh", "rona aa raha", "dil toot gaya", "ek dum sad",
        "kuch nahi ho raha", "akele hain", "bahut bura lag raha",
        "\u0995\u09be\u09a8\u09cd\u09a8\u09be \u09aa\u09be\u099a\u09cd\u099b\u09c7",  # কান্না পাচ্ছে
        "\u0996\u09c1\u09ac \u0995\u09b7\u09cd\u099f",                                # খুব কষ্ট
        "\u09ae\u09a8 \u0996\u09be\u09b0\u09be\u09aa",                                # মন খারাপ
        "\u09a6\u09c1\u0983\u0996",                                                    # দুঃখ
    ],
    "fear": [
        "bahut dar lag raha", "dar gaya", "dara hua", "bhoot jaisa",
        "itna darna", "andhera",
        "\u09ad\u09df \u09b2\u09be\u0997\u099b\u09c7",    # ভয় লাগছে
        "\u09ad\u09df \u09aa\u09be\u099a\u09cd\u099b\u09bf",  # ভয় পাচ্ছি
        "\u09ad\u09df\u0982\u0995\u09b0",                  # ভয়ংকর
    ],
    "disgust": [
        "chhi chhi", "yuck yaar", "kya bakwas hai", "bilkul pasand nahi",
        "ganda hai", "ulti aa rahi",
        "\u0998\u09c7\u09a8\u09cd\u09a8\u09be \u09b2\u09be\u0997\u099b\u09c7",  # ঘেন্না লাগছে
        "\u09ac\u09be\u099c\u09c7",                                                # বাজে
    ],
    "surprise": [
        "arre wah", "yaar kya baat", "sach mein", "aisa kaise", "oh my god yaar",
        "kya hua", "kitni badi baat",
        "\u0985\u09ac\u09be\u0995",       # অবাক
        "\u0986\u09b6\u09cd\u099a\u09b0\u09cd\u09af",  # আশ্চর্য
        "\u0985\u09ac\u09bf\u09b6\u09cd\u09ac\u09be\u09b8\u09cd\u09af",  # অবিশ্বাস্য
    ],
}

EMOTION_EMOJIS = {
    "joy":      "😄",
    "anger":    "😡",
    "sadness":  "😢",
    "fear":     "😱",
    "disgust":  "🤢",
    "surprise": "😲",
}

EMOTION_COLORS = {
    "joy":      "#FFD700",
    "anger":    "#FF4500",
    "sadness":  "#4169E1",
    "fear":     "#8A2BE2",
    "disgust":  "#228B22",
    "surprise": "#FF69B4",
}

MOON_PHASES = {
    "joy":      "🌕",
    "anger":    "🌑",
    "sadness":  "🌘",
    "fear":     "🌒",
    "disgust":  "🌓",
    "surprise": "🌙",
}

FAIRY_EMOJIS = {
    "joy":      "🧚‍♀️",
    "anger":    "🔥🧚",
    "sadness":  "🧚‍♂️",
    "fear":     "👻🧚",
    "disgust":  "🧚🍄",
    "surprise": "✨🧚‍♀️",
}

TAGLINES = {
    "joy":      "The stars smile with you tonight ✨",
    "anger":    "The cosmos feels your fire 🔥",
    "sadness":  "Even the moon weeps sometimes 🌧️",
    "fear":     "The void is vast, but you are not alone 🌒",
    "disgust":  "Some energies simply do not belong 🍄",
    "surprise": "The universe loves to astonish 💫",
}

EMOTIONS = list(EMOTION_KEYWORDS.keys())

# =============================================================================
# NLP UTILITIES
# =============================================================================

_lemmatizer = WordNetLemmatizer()
_sia        = SentimentIntensityAnalyzer()

try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = set()


def preprocess(text: str) -> str:
    """Lowercase → strip punctuation → tokenize → remove stopwords → lemmatize → rejoin."""
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words]
    return " ".join(tokens)


def translate_to_english(text: str) -> str:
    """Auto-detect source language and translate to English via deep_translator."""
    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        return result if result else text
    except Exception:
        return text


# =============================================================================
# SKLEARN ML MODEL  (TF-IDF + Logistic Regression, cached with joblib)
# =============================================================================

_MODEL_CACHE_PATH = os.path.join(tempfile.gettempdir(), "vibe_oracle_model.joblib")


def _build_training_corpus() -> pd.DataFrame:
    """
    Build a synthetic training DataFrame from keyword + phrase seeds.
    Returns a pandas DataFrame with columns ['text', 'label'].
    """
    # Sentence templates per keyword
    templates = [
        "I feel {w} today",
        "This makes me feel {w}",
        "Feeling so {w} right now",
        "I am completely {w}",
        "Everything feels {w}",
        "Such a {w} moment",
        "I cannot help but feel {w}",
        "It was truly {w}",
        "The {w} inside me is overwhelming",
        "So much {w}",
        "{w} is all I feel",
        "{w}",
    ]

    records = []
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            for tpl in templates:
                records.append({"text": tpl.format(w=kw), "label": emotion})
        # Also seed with multi-language phrases
        for phrase in MULTILANG_PHRASES.get(emotion, []):
            records.append({"text": phrase, "label": emotion})

    # Build DataFrame and shuffle with numpy for reproducibility
    df  = pd.DataFrame(records)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    df  = df.iloc[idx].reset_index(drop=True)
    return df


def _train_model():
    """Train TF-IDF + LogisticRegression pipeline; return (pipeline, label_encoder)."""
    df = _build_training_corpus()
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    X  = df["text"].apply(preprocess)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe, le


@st.cache_resource(show_spinner=False)
def get_model():
    """Return cached (pipeline, label_encoder). Train once; persist via joblib."""
    if os.path.exists(_MODEL_CACHE_PATH):
        try:
            return joblib.load(_MODEL_CACHE_PATH)
        except Exception:
            pass   # corrupt cache — retrain

    pipe, le = _train_model()
    try:
        joblib.dump((pipe, le), _MODEL_CACHE_PATH)
    except Exception:
        pass
    return pipe, le


# =============================================================================
# EMOTION DETECTION  (3-layer fusion)
# =============================================================================

def _rule_based_scores(raw_text: str, translated: str) -> dict:
    """Layer 1 + 2: multi-lang phrase hits (weight ×2) + keyword hits."""
    scores    = {e: 0.0 for e in EMOTIONS}
    raw_lower = raw_text.lower()

    # 1. Multi-language phrase detection on original text
    for emotion, phrases in MULTILANG_PHRASES.items():
        for phrase in phrases:
            if phrase.lower() in raw_lower:
                scores[emotion] += 2.0

    # 2. Keyword matching on translated + preprocessed text
    tokens        = set(preprocess(translated).split())
    translated_lc = translated.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in tokens or kw in translated_lc:
                scores[emotion] += 1.0

    return scores


def detect_emotion(raw_text: str) -> dict:
    """
    Multi-layer emotion detection → probability distribution over 6 emotions.

    Priority chain:
      1. Rule-based (phrase + keyword) — normalised if any signal
      2. ML model (TF-IDF + LogReg) probability vector
      3. VADER compound fallback when ML confidence is low
      4. Blend: 60% rule + 40% ML when rule has signal; 100% ML otherwise
    """
    # Translate once
    translated = translate_to_english(raw_text)

    # ── Layer 1 & 2: rule-based ───────────────────────────────────────────────
    rule_s     = _rule_based_scores(raw_text, translated)
    rule_total = sum(rule_s.values())

    # ── Layer 2: ML ───────────────────────────────────────────────────────────
    pipe, le      = get_model()
    processed     = preprocess(translated)
    ml_proba_arr  = pipe.predict_proba([processed])[0]          # numpy array
    ml_classes    = le.inverse_transform(np.arange(len(ml_proba_arr)))
    ml_scores     = {cls: float(ml_proba_arr[i])
                     for i, cls in enumerate(ml_classes)}
    for e in EMOTIONS:
        ml_scores.setdefault(e, 0.0)

    # ── Blend ────────────────────────────────────────────────────────────────
    if rule_total > 0:
        rule_proba = {e: rule_s[e] / rule_total for e in EMOTIONS}
        blended    = {e: 0.6 * rule_proba[e] + 0.4 * ml_scores[e] for e in EMOTIONS}
    else:
        blended = dict(ml_scores)
        # ── Layer 3: VADER safety-net when ML is uncertain ────────────────────
        top_conf = max(blended.values())
        if top_conf < 0.40:
            compound = _sia.polarity_scores(translated)["compound"]
            if compound >= 0.05:
                blended["joy"]     = blended.get("joy", 0)     + 0.50
            elif compound <= -0.05:
                blended["sadness"] = blended.get("sadness", 0) + 0.50
            total = sum(blended.values())
            blended = {e: v / total for e, v in blended.items()}

    # Final normalisation
    total = sum(blended.values())
    if total > 0:
        blended = {e: round(blended[e] / total, 4) for e in EMOTIONS}
    else:
        blended = {e: round(1.0 / len(EMOTIONS), 4) for e in EMOTIONS}

    return blended


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🌌 Vibe Oracle",
    page_icon="🔮",
    layout="centered",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Raleway:ital,wght@0,300;0,400;0,600;1,300&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #000010 !important;
    color: #f0e6ff !important;
    font-family: 'Raleway', sans-serif;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stMain"],
section.main { background: transparent !important; }

/* ── Cosmic nebula background ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse at 20% 30%, rgba(80,0,120,.35)  0%, transparent 55%),
        radial-gradient(ellipse at 80% 70%, rgba(0,50,120,.30)  0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(10,0,40,.90)   0%, transparent 100%);
    pointer-events: none; z-index: 0;
}

/* ── Star field ── */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed; inset: 0;
    background-image:
        radial-gradient(1px   1px   at 10% 15%,  white,                transparent),
        radial-gradient(1px   1px   at 25% 40%,  rgba(255,255,255,.8), transparent),
        radial-gradient(1.5px 1.5px at 40% 10%,  white,                transparent),
        radial-gradient(1px   1px   at 55% 60%,  rgba(255,255,255,.6), transparent),
        radial-gradient(1px   1px   at 70% 25%,  white,                transparent),
        radial-gradient(2px   2px   at 85% 45%,  rgba(255,255,255,.9), transparent),
        radial-gradient(1px   1px   at 15% 75%,  rgba(255,255,255,.7), transparent),
        radial-gradient(1.5px 1.5px at 60% 85%,  white,                transparent),
        radial-gradient(1px   1px   at 90% 80%,  rgba(255,255,255,.8), transparent),
        radial-gradient(1px   1px   at 35% 55%,  rgba(255,255,255,.5), transparent),
        radial-gradient(1px   1px   at 48% 33%,  rgba(200,180,255,.7), transparent),
        radial-gradient(1.5px 1.5px at 72% 58%,  rgba(180,220,255,.6), transparent);
    pointer-events: none; z-index: 0;
    animation: twinkle 4s ease-in-out infinite alternate;
}
@keyframes twinkle {
    0%   { opacity: .55; }
    100% { opacity: 1.0; }
}

/* ── Content wrapper ── */
.main-wrapper {
    position: relative; z-index: 1;
    text-align: center;
    padding: 1rem 0 2.5rem;
}

/* ── Title ── */
.oracle-title {
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(2rem, 6vw, 3.8rem);
    font-weight: 900;
    background: linear-gradient(135deg, #c084fc, #818cf8, #38bdf8, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: .04em;
    margin-bottom: .15rem;
    animation: titleGlow 3s ease-in-out infinite alternate;
}
@keyframes titleGlow {
    0%   { filter: drop-shadow(0 0  8px rgba(192,132,252,.6));  }
    100% { filter: drop-shadow(0 0 22px rgba(56,189,248,.85));  }
}

/* ── Subtitle ── */
.oracle-subtitle {
    font-size: 1.15rem; color: #c4b5fd;
    letter-spacing: .18em; margin-bottom: 1.6rem; font-weight: 300;
}

/* ── Floating fairy ── */
.fairy-float {
    font-size: 2.8rem; display: block;
    animation: fairyFloat 2.5s ease-in-out infinite;
    margin-bottom: .4rem;
}
@keyframes fairyFloat {
    0%,100% { transform: translateY(0)     rotate(-5deg); }
    50%      { transform: translateY(-14px) rotate(5deg);  }
}

/* ── Spinning moon ── */
.moon-spin {
    display: inline-block; font-size: 2rem;
    animation: moonSpin 9s linear infinite;
}
@keyframes moonSpin {
    from { transform: rotate(0deg);   }
    to   { transform: rotate(360deg); }
}

/* ── Aura card ── */
.aura-card {
    border-radius: 20px; padding: 2rem 1.5rem;
    margin: 1.6rem auto; max-width: 600px;
    backdrop-filter: blur(14px);
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.10);
    transition: box-shadow .5s ease;
}

/* ── Emotion name ── */
.emotion-name {
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(1.8rem, 5vw, 3rem);
    font-weight: 700; letter-spacing: .08em; margin: .4rem 0;
    animation: emotionPulse 2s ease-in-out infinite alternate;
}
@keyframes emotionPulse {
    0%   { filter: brightness(1);                                    }
    100% { filter: brightness(1.35) drop-shadow(0 0 14px currentColor); }
}

.emotion-emoji-big {
    font-size: 3.5rem; display: block; margin: .3rem 0;
    animation: emojiBounce 1.2s ease-in-out infinite;
}
@keyframes emojiBounce {
    0%,100% { transform: scale(1);   }
    50%      { transform: scale(1.2); }
}

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    font-size: .78rem; font-weight: 600; letter-spacing: .12em;
    padding: .25rem .75rem; border-radius: 999px;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.15);
    color: #e2d9f3; margin-top: .5rem;
    font-family: 'Raleway', sans-serif;
}

/* ── Emotion breakdown bars ── */
.bar-row {
    display: flex; align-items: center;
    gap: 10px; margin: 7px 0;
    font-family: 'Raleway', sans-serif;
}
.bar-label {
    width: 95px; text-align: right;
    font-size: .82rem; color: #e2d9f3;
    font-weight: 600; text-transform: capitalize;
}
.bar-track {
    flex: 1; height: 16px;
    background: rgba(255,255,255,.07);
    border-radius: 8px; overflow: hidden;
}
.bar-fill {
    height: 100%; border-radius: 8px;
    animation: barGrow 1.3s cubic-bezier(.23,1,.32,1) forwards;
}
@keyframes barGrow {
    from { width: 0%;             }
    to   { width: var(--bar-w); }
}
.bar-pct {
    width: 44px; font-size: .8rem;
    color: #c4b5fd; font-weight: 600;
}

/* ── Section label ── */
.section-label {
    font-family: 'Cinzel Decorative', serif;
    font-size: .82rem; letter-spacing: .2em;
    color: #a78bfa; text-transform: uppercase; margin-bottom: .8rem;
}

/* ── Divider ── */
.mystic-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,.5), transparent);
    margin: 1.5rem 0;
}

/* ── Warning ── */
.warn-msg {
    color: #fbbf24; font-size: 1rem;
    padding: .8rem 1.2rem;
    border: 1px solid rgba(251,191,36,.3); border-radius: 10px;
    background: rgba(251,191,36,.08); margin-top: 1rem;
}

/* ── Streamlit widget overrides ── */
textarea {
    background: rgba(255,255,255,.05) !important;
    color: #f0e6ff !important;
    border: 1px solid rgba(168,85,247,.4) !important;
    border-radius: 12px !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 1rem !important;
}
textarea:focus {
    border-color: rgba(168,85,247,.9) !important;
    box-shadow: 0 0 20px rgba(168,85,247,.3) !important;
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #0ea5e9) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; padding: .7rem 2.5rem !important;
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 1.05rem !important; letter-spacing: .05em !important;
    cursor: pointer !important; transition: all .3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,.5) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,.7) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LAYOUT
# =============================================================================

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="oracle-title">🌌 Vibe Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="oracle-subtitle">Speak your vibe 🌙</p>', unsafe_allow_html=True)

# ── Warm the model once (cached after first run) ──────────────────────────────
with st.spinner("🔭 Aligning the cosmic model…"):
    _pipe, _le = get_model()

# ── Text input ────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="",
    placeholder="Type anything… in any language 🔮",
    height=130,
    key="vibe_input",
    label_visibility="collapsed",
)

c1, c2, c3 = st.columns([2, 1.5, 2])
with c2:
    reveal = st.button("🔮 Reveal the Vibe", use_container_width=True)

# ── Detection & output ────────────────────────────────────────────────────────
if reveal:
    if not user_input.strip():
        st.markdown(
            '<div class="warn-msg">'
            '⚠️ Please enter some text to reveal your vibe…'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("✨ Reading the cosmic vibrations…"):
            scores = detect_emotion(user_input)

        # Use pandas Series for ordering / max lookup
        score_series = pd.Series(scores, dtype=float).sort_values(ascending=False)
        dominant     = str(score_series.idxmax())
        dominant_pct = int(round(score_series[dominant] * 100))

        color = EMOTION_COLORS[dominant]
        emoji = EMOTION_EMOJIS[dominant]
        fairy = FAIRY_EMOJIS[dominant]
        moon  = MOON_PHASES[dominant]
        tag   = TAGLINES[dominant]

        # ── Aura card ─────────────────────────────────────────────────────────
        glow = (
            f"0 0 60px {color}55, "
            f"0 0 120px {color}22, "
            f"inset 0 0 40px {color}11"
        )
        st.markdown(f"""
        <div class="aura-card" style="box-shadow:{glow}; border-color:{color}44;">
            <span class="fairy-float">{fairy}</span>
            <span class="moon-spin">{moon}</span>
            <p class="section-label">Your Dominant Vibe</p>
            <span class="emotion-emoji-big">{emoji}</span>
            <p class="emotion-name" style="color:{color};">{dominant.upper()}</p>
            <p style="color:#c4b5fd; font-size:.95rem; margin-top:.2rem;
                      font-family:'Raleway',sans-serif;">
                {fairy} &nbsp; {moon} &nbsp; {emoji}
            </p>
            <span class="conf-badge">✦ Confidence: {dominant_pct}% ✦</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Emotion breakdown bars ────────────────────────────────────────────
        st.markdown('<hr class="mystic-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label" style="text-align:center;">'
            '✦ Emotion Breakdown ✦</p>',
            unsafe_allow_html=True,
        )

        # Iterate emotions in descending order (pandas sort already done)
        for emotion in score_series.index.tolist():
            pct = int(round(score_series[emotion] * 100))
            ec  = EMOTION_COLORS[emotion]
            ee  = EMOTION_EMOJIS[emotion]

            if pct == 100:
                fill     = f"linear-gradient(90deg, {ec}, #fff8)"
                glow_bar = f"0 0 10px {ec}"
            else:
                fill     = f"linear-gradient(90deg, {ec}cc, {ec}44)"
                glow_bar = f"0 0 6px {ec}88"

            st.markdown(f"""
            <div class="bar-row">
                <span class="bar-label">{ee} {emotion}</span>
                <div class="bar-track">
                    <div class="bar-fill"
                         style="--bar-w:{pct}%;
                                width:{pct}%;
                                background:{fill};
                                box-shadow:{glow_bar};">
                    </div>
                </div>
                <span class="bar-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Mystical tagline ──────────────────────────────────────────────────
        st.markdown(f"""
        <p style="text-align:center; margin-top:1.2rem;
                  font-style:italic; color:#a78bfa;
                  font-size:.95rem; letter-spacing:.06em;
                  font-family:'Raleway',sans-serif;">
            {tag}
        </p>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)   # close .main-wrapper

import streamlit as st
import re
import difflib
import nltk
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Celestial Emotion Oracle", page_icon="🌙", layout="centered")

# ---------------- NLTK SETUP ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not","no","very","never"}
STOP_WORDS = STOP_WORDS.difference(KEEP)

# ---------------- EMOTION SETTINGS ----------------
EMO_DICT = {
    "joy": {"happy","love","awesome","amazing","great","fantastic","smile"},
    "sadness": {"sad","cry","lonely","depressed","heartbroken"},
    "anger": {"angry","hate","furious","mad","annoyed"},
    "fear": {"scared","panic","terrified","afraid","nervous"},
    "disgust": {"gross","nasty","eww"},
    "surprise": {"wow","unexpected","omg","shocked"}
}

EMOTION_AURA = {
    "joy": "#facc15",
    "sadness": "#60a5fa",
    "anger": "#ef4444",
    "fear": "#a78bfa",
    "disgust": "#22c55e",
    "surprise": "#f472b6"
}

EMOTION_MOON = {
    "joy": "🌕",
    "sadness": "🌑",
    "anger": "🌖",
    "fear": "🌘",
    "disgust": "🌒",
    "surprise": "🌔"
}

ALL_WORDS = sorted(set().union(*EMO_DICT.values()))

# ---------------- NLP ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]

def emotion_scores(text):
    tokens = preprocess(text)
    scores = {e:1 for e in EMO_DICT}  # baseline to avoid neutral

    for tok in tokens:
        for emo,words in EMO_DICT.items():
            if tok in words:
                scores[emo]+=2

        close = difflib.get_close_matches(tok, ALL_WORDS, n=1, cutoff=0.8)
        if close:
            for emo,words in EMO_DICT.items():
                if close[0] in words:
                    scores[emo]+=1

    vader = sia.polarity_scores(text)["compound"]
    if vader > 0:
        scores["joy"] += abs(vader)*2
    elif vader < 0:
        scores["sadness"] += abs(vader)*2

    total = sum(scores.values())
    probs = {k:round(v/total,2) for k,v in scores.items()}
    return probs

def predict(text):
    probs = emotion_scores(text)
    best = max(probs, key=probs.get)
    return best, probs

# ---------------- DEFAULT THEME ----------------
aura_color = "#a78bfa"
moon_symbol = "🌙"

# ---------------- UI STYLE ----------------
st.markdown(f"""
<style>

.stApp {{
    background: radial-gradient(circle at top, {aura_color}22, #0f172a 60%);
    color: #e2e8f0;
    transition: background 1.5s ease;
}}

.block-container {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2rem;
    box-shadow: 0 0 50px {aura_color};
}}

.moon {{
    font-size: 90px;
    text-align:center;
    animation: floatMoon 4s ease-in-out infinite;
}}

@keyframes floatMoon {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-15px); }}
    100% {{ transform: translateY(0px); }}
}}

.title {{
    text-align:center;
    font-size:45px;
    font-weight:700;
    margin-bottom:10px;
}}

textarea {{
    background-color: rgba(30,27,75,0.7) !important;
    color: white !important;
    border-radius: 20px !important;
    border: 1px solid {aura_color} !important;
}}

.stButton>button {{
    background: linear-gradient(90deg,{aura_color},#7c3aed);
    border-radius: 30px;
    padding: 12px 30px;
    font-size: 17px;
    color: white;
    border: none;
    box-shadow: 0 0 25px {aura_color};
}}

.stProgress > div > div {{
    background: linear-gradient(90deg,{aura_color},#ffffff);
}}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌙 Celestial Emotion Oracle ✨</div>', unsafe_allow_html=True)
st.markdown(f'<div class="moon">{moon_symbol}</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area("Whisper your thoughts... (English / Hindi / Bengali supported)", height=150)

if st.button("🔮 Reveal Emotion"):

    if not text.strip():
        st.warning("The oracle awaits your words...")
    else:
        with st.spinner("Reading celestial vibrations..."):

            translated = GoogleTranslator(source='auto', target='en').translate(text)
            emotion, probs = predict(translated)

            aura_color = EMOTION_AURA[emotion]
            moon_symbol = EMOTION_MOON[emotion]

        # Re-render theme with emotion color
        st.markdown(f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top, {aura_color}33, #0f172a 60%);
        }}
        .block-container {{
            box-shadow: 0 0 60px {aura_color};
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="moon">{moon_symbol}</div>', unsafe_allow_html=True)

        st.subheader(f"✨ Dominant Emotion: {emotion.upper()}")

        st.markdown("### 🌌 Emotional Aura Distribution")

        for emo,val in probs.items():
            st.write(f"{emo.capitalize()} — {int(val*100)}%")
            st.progress(val)

        st.success("The moon has shifted with your soul.")

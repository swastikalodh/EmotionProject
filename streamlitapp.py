import streamlit as st
import re
import difflib
import nltk
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌌 Vibe Oracle",
    page_icon="🌙",
    layout="centered"
)

# ---------------- NLTK SETUP ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not","no","never","very"}
STOP_WORDS = STOP_WORDS.difference(KEEP)

# ---------------- EMOTION DATA ----------------
EMO_DICT = {
    "joy": {"happy","love","awesome","amazing","great","fantastic","smile"},
    "anger": {"angry","hate","furious","mad","annoyed"},
    "sadness": {"sad","cry","lonely","depressed","heartbroken"},
    "fear": {"scared","panic","terrified","afraid","nervous"},
    "disgust": {"gross","nasty","eww"},
    "surprise": {"wow","unexpected","omg","shocked"}
}

EMO_EMOJI = {
    "joy":"😊",
    "anger":"😡",
    "sadness":"😢",
    "fear":"😨",
    "disgust":"🤢",
    "surprise":"😲"
}

EMO_COLOR = {
    "joy":"#facc15",
    "anger":"#ef4444",
    "sadness":"#60a5fa",
    "fear":"#a78bfa",
    "disgust":"#22c55e",
    "surprise":"#f472b6"
}

EMO_MOON = {
    "joy":"🌕",
    "anger":"🌖",
    "sadness":"🌑",
    "fear":"🌘",
    "disgust":"🌒",
    "surprise":"🌔"
}

# ---------------- NLP ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]

def detect_emotion(text):
    tokens = preprocess(text)
    scores = {e:0 for e in EMO_DICT}

    for tok in tokens:
        for emo, words in EMO_DICT.items():
            if tok in words:
                scores[emo] += 1

    non_zero = {k:v for k,v in scores.items() if v > 0}

    if len(non_zero) == 1:
        emo = list(non_zero.keys())[0]
        return emo, {emo: 1.0}

    if len(non_zero) > 1:
        total = sum(non_zero.values())
        probs = {k: round(v/total, 2) for k,v in non_zero.items()}
        best = max(probs, key=probs.get)
        return best, probs

    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0:
        return "joy", {"joy": 1.0}
    else:
        return "sadness", {"sadness": 1.0}

# ---------------- UI BASE STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: black;
    background-image: url("https://www.transparenttextures.com/patterns/stardust.png");
    color: #e5e7eb;
}

.title {
    text-align:center;
    font-size:46px;
    font-weight:700;
}

textarea {
    background: rgba(20,20,40,0.7) !important;
    color: white !important;
    border-radius: 20px !important;
}

/* Tarot Flip */
.flip-card {
    perspective:1000px;
}
.flip-inner {
    position:relative;
    transition: transform 1s;
    transform-style:preserve-3d;
}
.flip-card.flipped .flip-inner {
    transform: rotateY(180deg);
}

/* Animated Bars */
.bar-container {
    background:#1e1b4b;
    border-radius:20px;
    margin-bottom:8px;
}
.bar-fill {
    height:20px;
    border-radius:20px;
    animation: growBar 1.5s ease forwards;
}
@keyframes growBar {
    from { width:0%; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌌 Vibe Oracle ✨</div>', unsafe_allow_html=True)
st.caption("Drop a thought. Let the universe read your vibe.")

# ---------------- INPUT ----------------
text = st.text_area("✨ Speak your vibe (English / Hindi / Bengali)", height=150)

if st.button("🔮 Reveal the Vibe"):

    if not text.strip():
        st.warning("Say *something*… the stars are listening 👀")
    else:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        emotion, probs = detect_emotion(translated)

        color = EMO_COLOR[emotion]
        moon = EMO_MOON[emotion]
        emoji = EMO_EMOJI[emotion]
        confidence = max(probs.values())

        pulse_speed = 6 - (confidence * 4)

        # ⭐ Dynamic Effects
        st.markdown(f"""
        <style>

        .stApp::before {{
            content:"";
            position:fixed;
            width:100%;
            height:100%;
            background: radial-gradient(circle, {color}22, transparent 70%);
            animation: pulse {pulse_speed}s infinite alternate;
            z-index:-1;
        }}

        @keyframes pulse {{
            from {{ opacity:0.3; }}
            to {{ opacity:0.7; }}
        }}

        .block-container {{
            box-shadow: 0 0 40px {color};
            animation: breathe 3s ease-in-out infinite;
            border-radius:25px;
        }}

        @keyframes breathe {{
            0% {{ box-shadow:0 0 30px {color}; }}
            50% {{ box-shadow:0 0 80px {color}; }}
            100% {{ box-shadow:0 0 30px {color}; }}
        }}

        .moon {{
            font-size:90px;
            text-align:center;
            filter: brightness({0.6 + confidence});
        }}

        </style>
        """, unsafe_allow_html=True)

        # 🔮 Tarot Flip Reveal
        st.markdown('<div class="flip-card flipped"><div class="flip-inner">', unsafe_allow_html=True)

        st.markdown(f'<div class="moon">{moon}</div>', unsafe_allow_html=True)
        st.subheader(f"{emoji} Dominant Vibe: **{emotion.upper()}**")

        st.markdown("### 🌌 Vibe Breakdown")

        for emo,val in probs.items():
            width = int(val*100)
            st.markdown(f"""
            <div class="bar-container">
                <div class="bar-fill" style="width:{width}%; background:{color};"></div>
            </div>
            <small>{EMO_EMOJI[emo]} {emo.capitalize()} — {width}%</small>
            """, unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        st.success("✨ Vibe successfully decoded by the cosmos.")
        

import streamlit as st
import re
import nltk
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Celestial Emotion Oracle", page_icon="🌙", layout="centered")

# ---------------- NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not","no","very","never"}
STOP_WORDS = STOP_WORDS.difference(KEEP)

# ---------------- EMOTIONS ----------------
EMO_DICT = {
    "joy": {"happy","love","awesome","amazing","great","fantastic","smile"},
    "sadness": {"sad","cry","lonely","depressed","heartbroken"},
    "anger": {"angry","hate","furious","mad","annoyed"},
    "fear": {"scared","panic","terrified","afraid","nervous"},
    "disgust": {"gross","nasty","eww"},
    "surprise": {"wow","unexpected","omg","shocked"}
}

EMOTION_AURA = {
    "joy": "#ffd700",
    "sadness": "#60a5fa",
    "anger": "#ff3b3b",
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

    non_zero = {k:v for k,v in scores.items() if v>0}

    if len(non_zero) == 1:
        emo = list(non_zero.keys())[0]
        return emo, {emo:1.0}

    if len(non_zero) > 1:
        total = sum(non_zero.values())
        probs = {k:round(v/total,2) for k,v in non_zero.items()}
        best = max(probs, key=probs.get)
        return best, probs

    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0:
        return "joy", {"joy":1.0}
    else:
        return "sadness", {"sadness":1.0}

# ---------------- BASE STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#0f0c29,#302b63,#24243e);
    background-size:400% 400%;
    animation: gradientShift 20s ease infinite;
    color:white;
}

@keyframes gradientShift {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.moon {
    font-size:90px;
    text-align:center;
    transition: all 1s ease;
}

.block-container {
    border-radius:25px;
    padding:2rem;
    transition: all 1s ease;
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
.flip-front, .flip-back {
    backface-visibility:hidden;
}
.flip-back {
    transform: rotateY(180deg);
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🌙 Celestial Emotion Oracle ✨")

# ---------------- INPUT ----------------
text = st.text_area("Whisper your thoughts... (English / Hindi / Bengali supported)", height=150)

if st.button("🔮 Reveal Emotion"):

    if not text.strip():
        st.warning("The oracle awaits your words...")
    else:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        emotion, probs = detect_emotion(translated)

        aura_color = EMOTION_AURA[emotion]
        moon_symbol = EMOTION_MOON[emotion]
        confidence = max(probs.values())

        # ⭐ Pulsing stars based on confidence
        pulse_speed = 6 - (confidence * 4)

        st.markdown(f"""
        <style>
        .stApp::before {{
            content:"";
            position:fixed;
            top:0;
            left:0;
            width:100%;
            height:100%;
            background:url("https://www.transparenttextures.com/patterns/stardust.png");
            opacity:0.4;
            animation: starPulse {pulse_speed}s ease-in-out infinite;
            z-index:-1;
        }}

        @keyframes starPulse {{
            0%{{opacity:0.2;}}
            50%{{opacity:0.7;}}
            100%{{opacity:0.2;}}
        }}

        .block-container {{
            box-shadow:0 0 70px {aura_color};
            animation: breathe 4s ease-in-out infinite;
        }}

        @keyframes breathe {{
            0%{{box-shadow:0 0 30px {aura_color};}}
            50%{{box-shadow:0 0 90px {aura_color};}}
            100%{{box-shadow:0 0 30px {aura_color};}}
        }}

        .moon {{
            filter: brightness({0.6 + confidence});
        }}
        </style>
        """, unsafe_allow_html=True)

        # 🔮 Tarot Flip Reveal
        st.markdown('<div class="flip-card flipped"><div class="flip-inner">', unsafe_allow_html=True)

        st.markdown(f'<div class="moon">{moon_symbol}</div>', unsafe_allow_html=True)
        st.subheader(f"✨ Dominant Emotion: {emotion.upper()}")

        st.markdown("### 🌌 Emotional Aura Distribution")

        for emo,val in probs.items():
            percent = int(val*100)
            st.progress(val)
            st.write(f"{emo.capitalize()} — {percent}%")

        st.markdown('</div></div>', unsafe_allow_html=True)

        st.success("The stars have aligned with your emotion.")

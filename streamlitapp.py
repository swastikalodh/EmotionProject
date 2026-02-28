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

ALL_WORDS = sorted(set().union(*EMO_DICT.values()))

# ---------------- NLP ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]

def emotion_scores(text):
    tokens = preprocess(text)
    scores = {e:0 for e in EMO_DICT}

    for tok in tokens:
        for emo,words in EMO_DICT.items():
            if tok in words:
                scores[emo]+=1

    non_zero = {k:v for k,v in scores.items() if v>0}

    # Single clear emotion → 100%
    if len(non_zero) == 1:
        emo = list(non_zero.keys())[0]
        return emo, {emo:1.0}

    # Mixed emotions
    if len(non_zero) > 1:
        total = sum(non_zero.values())
        probs = {k:round(v/total,2) for k,v in non_zero.items()}
        best = max(probs, key=probs.get)
        return best, probs

    # Sentiment fallback
    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0:
        return "joy", {"joy":1.0}
    else:
        return "sadness", {"sadness":1.0}

def predict(text):
    return emotion_scores(text)

# ---------------- DEFAULT THEME ----------------
aura_color = "#a78bfa"
moon_symbol = "🌙"

# ---------------- UI STYLE ----------------
st.markdown(f"""
<style>

/* Animated mystical gradient */
.stApp {{
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color: #e2e8f0;
}}

@keyframes gradientShift {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

/* ⭐ Pulsating Star Layer */
.stApp::before {{
    content:"";
    position:fixed;
    top:0;
    left:0;
    width:100%;
    height:100%;
    background:url("https://www.transparenttextures.com/patterns/stardust.png");
    opacity:0.35;
    animation: starPulse 6s ease-in-out infinite;
    z-index:-1;
}}

@keyframes starPulse {{
    0% {{ opacity:0.2; }}
    50% {{ opacity:0.6; }}
    100% {{ opacity:0.2; }}
}}

/* Glass mystical card */
.block-container {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    border-radius: 25px;
    padding: 2rem;
    box-shadow: 0 0 50px {aura_color};
    animation: breathe 4s ease-in-out infinite;
}}

@keyframes breathe {{
    0% {{ box-shadow:0 0 25px {aura_color}; }}
    50% {{ box-shadow:0 0 70px {aura_color}; }}
    100% {{ box-shadow:0 0 25px {aura_color}; }}
}}

/* Floating moon */
.moon {{
    font-size: 90px;
    text-align:center;
    animation: floatMoon 6s ease-in-out infinite;
}}

@keyframes floatMoon {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-12px); }}
    100% {{ transform: translateY(0px); }}
}}

.title {{
    text-align:center;
    font-size:45px;
    font-weight:700;
}

/* Animated emotion bars */
.bar-container {{
    background:#1e1b4b;
    border-radius:20px;
    margin-bottom:8px;
}}

.bar-fill {{
    height:20px;
    border-radius:20px;
    animation: growBar 1.5s ease forwards;
}}

@keyframes growBar {{
    from {{ width:0%; }}
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
            confidence = max(probs.values())

        # Dynamic glow + moon brightness scaling
        st.markdown(f"""
        <style>
        .block-container {{
            box-shadow:0 0 70px {aura_color};
        }}
        .moon {{
            filter: brightness({0.6 + confidence});
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="moon">{moon_symbol}</div>', unsafe_allow_html=True)

        st.subheader(f"✨ Dominant Emotion: {emotion.upper()}")

        st.markdown("### 🌌 Emotional Aura Distribution")

        for emo,val in probs.items():
            width = int(val*100)
            st.markdown(f"""
            <div class="bar-container">
                <div class="bar-fill" style="width:{width}%; background:{aura_color};"></div>
            </div>
            <small>{emo.capitalize()} — {width}%</small>
            """, unsafe_allow_html=True)

        st.success("The moon has shifted with your soul.")

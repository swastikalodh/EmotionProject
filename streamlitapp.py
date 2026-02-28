import streamlit as st
import re
import difflib
import nltk
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="✨ Vibe Reader ✨", page_icon="🌌", layout="centered")

# ---------------- NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# ---------------- EMOTION DATA ----------------
EMO_DICT = {
    "joy": {"happy","love","awesome","great","amazing","smile"},
    "anger": {"hate","angry","furious","mad","annoyed"},
    "sadness": {"sad","cry","lonely","depressed"},
    "fear": {"scared","afraid","panic","terrified"},
    "disgust": {"gross","nasty","eww"},
    "surprise": {"wow","unexpected","omg"}
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

    # If only one emotion appears → 100%
    non_zero = [k for k,v in scores.items() if v>0]

    if len(non_zero) == 1:
        return non_zero[0], {non_zero[0]:1.0}

    # Mixed emotions
    total = sum(scores.values())
    if total == 0:
        # fallback using sentiment
        compound = sia.polarity_scores(text)["compound"]
        if compound > 0:
            return "joy", {"joy":1.0}
        else:
            return "sadness", {"sadness":1.0}

    probs = {k:round(v/total,2) for k,v in scores.items() if v>0}
    best = max(probs, key=probs.get)
    return best, probs

# ---------------- UI ----------------
text = st.text_area("✨ Drop your vibe here (Hindi / Bengali / English works)", height=150)

if st.button("🔮 Reveal The Vibe"):

    if not text.strip():
        st.warning("Say something dramatic at least 😌")
    else:

        translated = GoogleTranslator(source='auto', target='en').translate(text)
        emotion, probs = emotion_scores(translated)

        color = EMO_COLOR[emotion]
        moon = EMO_MOON[emotion]
        emoji = EMO_EMOJI[emotion]
        confidence = list(probs.values())[0] if len(probs)==1 else max(probs.values())

        # ---------------- DYNAMIC COSMIC UI ----------------
        st.markdown(f"""
        <style>
        .stApp {{
            background: black;
            overflow:hidden;
        }}

        /* Star Field */
        .stars {{
            position: fixed;
            width: 100%;
            height: 100%;
            background: url("https://www.transparenttextures.com/patterns/stardust.png");
            animation: pulseStars {5 - confidence*3}s infinite alternate;
            opacity:{0.3 + confidence*0.5};
            z-index:-1;
        }}

        @keyframes pulseStars {{
            from {{ opacity:0.2; }}
            to {{ opacity:0.6; }}
        }}

        /* Breathing Aura */
        .aura {{
            box-shadow: 0 0 60px {color};
            animation: breathe 3s ease-in-out infinite;
            border-radius:20px;
            padding:20px;
        }}

        @keyframes breathe {{
            0% {{ box-shadow:0 0 40px {color}; }}
            50% {{ box-shadow:0 0 80px {color}; }}
            100% {{ box-shadow:0 0 40px {color}; }}
        }}

        /* Tarot Flip */
        .flip-card {{
            perspective: 1000px;
        }}

        .flip-inner {{
            position: relative;
            width: 100%;
            text-align: center;
            transition: transform 1s;
            transform-style: preserve-3d;
            transform: rotateY(180deg);
        }}

        .flip-front, .flip-back {{
            backface-visibility: hidden;
            position:absolute;
            width:100%;
        }}

        .flip-back {{
            transform: rotateY(180deg);
        }}

        /* Moon brightness */
        .moon {{
            font-size:80px;
            filter: brightness({0.7 + confidence});
        }}
        </style>

        <div class="stars"></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="aura">', unsafe_allow_html=True)

        st.markdown(f'<div class="moon">{moon}</div>', unsafe_allow_html=True)

        st.markdown(f"## {emoji} Dominant Vibe: **{emotion.upper()}**")

        st.markdown("### 🎴 Cosmic Breakdown")

        for emo,val in probs.items():
            st.write(f"{EMO_EMOJI[emo]} {emo.capitalize()} — {int(val*100)}%")

        st.markdown("</div>", unsafe_allow_html=True)

        st.success("✨ The universe has delivered your vibe.")

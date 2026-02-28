import streamlit as st
import re
import nltk
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🌌 Vibe Oracle", page_icon="🌙", layout="centered")

# ---------------- NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

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

EMO_MUSIC = {
    "joy":"https://cdn.pixabay.com/download/audio/2022/03/15/audio_2f8d0b0e1e.mp3",
    "anger":"https://cdn.pixabay.com/download/audio/2021/09/15/audio_abc123.mp3",
    "sadness":"https://cdn.pixabay.com/download/audio/2022/02/10/audio_def456.mp3",
    "fear":"https://cdn.pixabay.com/download/audio/2021/10/05/audio_xyz789.mp3",
    "disgust":"https://cdn.pixabay.com/download/audio/2021/08/01/audio_ghj111.mp3",
    "surprise":"https://cdn.pixabay.com/download/audio/2021/09/10/audio_qwe222.mp3"
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
                scores[emo]+=1

    non_zero = {k:v for k,v in scores.items() if v>0}

    if len(non_zero)==1:
        emo=list(non_zero.keys())[0]
        return emo,{emo:1.0}

    if len(non_zero)>1:
        total=sum(non_zero.values())
        probs={k:round(v/total,2) for k,v in non_zero.items()}
        best=max(probs,key=probs.get)
        return best,probs

    compound=sia.polarity_scores(text)["compound"]
    if compound>=0:
        return "joy",{"joy":1.0}
    else:
        return "sadness",{"sadness":1.0}

# ---------------- UI BASE ----------------
st.markdown("""
<style>
.stApp {
    background: black;
    background-image: url("https://www.transparenttextures.com/patterns/stardust.png");
    color:white;
}

/* rotating moon */
.moon {
    font-size:90px;
    text-align:center;
    animation: rotateMoon 20s linear infinite;
}
@keyframes rotateMoon {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* live aura while typing */
textarea:focus {
    box-shadow: 0 0 30px #a78bfa !important;
    transition:0.3s;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌌 Vibe Oracle ✨")

# --------- FORM (ENTER KEY WORKS) ----------
with st.form(key="vibe_form"):
    user_text = st.text_area("Type your vibe here (Press Enter to reveal)", height=150)
    submitted = st.form_submit_button("🔮 Reveal")

if submitted and user_text.strip():

    translated = GoogleTranslator(source='auto', target='en').translate(user_text)
    emotion, probs = detect_emotion(translated)

    color = EMO_COLOR[emotion]
    moon = EMO_MOON[emotion]
    emoji = EMO_EMOJI[emotion]
    confidence = max(probs.values())

    # -------- dynamic constellation --------
    constellation = "✨ ✦ ✧ ✨" if emotion=="joy" else \
                    "✹ ✸ ✹" if emotion=="anger" else \
                    "⋆ ⋆ ⋆" if emotion=="sadness" else \
                    "✦ ✧ ✦" if emotion=="fear" else \
                    "✧ ✧" if emotion=="disgust" else \
                    "✶ ✷ ✶"

    st.markdown(f"""
    <style>
    .block-container {{
        box-shadow:0 0 60px {color};
        border-radius:20px;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="moon">{moon}</div>', unsafe_allow_html=True)

    st.markdown(f"## {emoji} Dominant Vibe: **{emotion.upper()}**")
    st.markdown(f"### 🌌 Constellation: {constellation}")

    for emo,val in probs.items():
        st.write(f"{EMO_EMOJI[emo]} {emo.capitalize()} — {int(val*100)}%")

    # emotion-based music
    st.audio(EMO_MUSIC[emotion], autoplay=True)

    st.success("✨ Cosmic vibe revealed.")

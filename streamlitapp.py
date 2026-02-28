import streamlit as st
import re
import difflib
import nltk
import pandas as pd
from googletrans import Translator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Emotion Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CSS DESIGN ----------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#BB86FC;
    animation: fadeIn 2s ease-in-out;
}

.subtitle {
    text-align:center;
    color:#cccccc;
    margin-bottom:30px;
}

textarea {
    border-radius:15px !important;
    background-color:#1e1e2f !important;
    color:white !important;
}

.stButton>button {
    background: linear-gradient(90deg,#8e2de2,#4a00e0);
    color:white;
    border-radius:20px;
    padding:10px 25px;
    font-size:16px;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.05);
    background: linear-gradient(90deg,#4a00e0,#8e2de2);
}

@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✨ AI Emotion Analyzer ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multilingual • Animated • Intelligent</div>', unsafe_allow_html=True)

# ---------------- DOWNLOAD NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
translator = Translator()

STOP_WORDS = set(stopwords.words("english"))
KEEP = {"not","no","very","never"}
STOP_WORDS = STOP_WORDS.difference(KEEP)

# ---------------- EMOTION DICTIONARY ----------------
EMO_DICT = {
    "joy": {"happy","love","awesome","amazing","great","fantastic","smile"},
    "sadness": {"sad","cry","lonely","depressed","heartbroken"},
    "anger": {"angry","hate","idiot","furious","mad"},
    "fear": {"scared","panic","terrified","afraid"},
    "disgust": {"gross","vomit","nasty","eww"},
    "surprise": {"wow","omg","unexpected"}
}

EMO_EMOJI = {
    "joy":"😊","sadness":"😢","anger":"😠",
    "fear":"😨","disgust":"🤢","surprise":"😲",
    "neutral":"🤖"
}

EMO_GIF = {
    "joy":"assets/happy.gif",
    "sadness":"assets/sad.gif",
    "anger":"assets/angry.gif",
    "fear":"assets/fear.gif",
    "disgust":"assets/disgust.gif",
    "surprise":"assets/surprise.gif",
    "neutral":"assets/robot.gif"
}

ALL_WORDS = sorted(set().union(*EMO_DICT.values()))

# ---------------- NLP FUNCTIONS ----------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    clean=[]
    for t in tokens:
        if t not in STOP_WORDS:
            clean.append(lemmatizer.lemmatize(t))
    return clean

def emotion_scores(text):
    tokens = preprocess(text)
    scores = {e:0 for e in EMO_DICT}

    for tok in tokens:
        for emo,words in EMO_DICT.items():
            if tok in words:
                scores[emo]+=2

        close = difflib.get_close_matches(tok, ALL_WORDS, n=1, cutoff=0.8)
        if close:
            for emo,words in EMO_DICT.items():
                if close[0] in words:
                    scores[emo]+=1

    total = sum(scores.values())+0.0001
    probs = {k:round(v/total,2) for k,v in scores.items()}
    return probs

def vader_predict(text):
    c = sia.polarity_scores(text)["compound"]
    if c>0.4: return "joy"
    if c<-0.4: return "sadness"
    return None

def predict(text):
    probs = emotion_scores(text)
    best = max(probs, key=probs.get)

    if probs[best]>0:
        return best,probs

    v = vader_predict(text)
    if v:
        return v,{v:1.0}

    return "neutral",{"neutral":1.0}

# ---------------- UI ----------------

language = st.selectbox(
    "🌍 Choose Input Language",
    ["English", "Hindi", "Bengali"]
)

text = st.text_area("✍ Enter your text", height=120)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Analyze Emotion"):

    if not text.strip():
        st.warning("Please type something")
    else:
        with st.spinner("Analyzing emotions... 🤖"):
            
            # Translate if needed
            if language != "English":
                translated = translator.translate(text, dest="en")
                text = translated.text
                st.info(f"Translated to English: {text}")

            emotion, probs = predict(text)
            st.session_state.history.append(emotion)

        st.subheader(f"{EMO_EMOJI.get(emotion)} Detected Emotion: **{emotion.upper()}**")

        # Show GIF reaction
        gif_path = EMO_GIF.get(emotion)
        if gif_path:
            st.image(gif_path, width=250)

        st.divider()
        st.markdown("### 📊 Confidence Levels")

        for emo,val in probs.items():
            st.write(f"{emo.capitalize()} — {int(val*100)}%")
            st.progress(val)

        st.success("✨ Analysis Complete!")

# ---------------- HISTORY CHART ----------------
if st.session_state.history:
    st.divider()
    st.markdown("### 📈 Emotion History")
    df = pd.DataFrame(st.session_state.history, columns=["Emotion"])
    st.bar_chart(df["Emotion"].value_counts())

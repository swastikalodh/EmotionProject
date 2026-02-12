import streamlit as st
import re
import difflib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- SET PROFESSIONAL THEME ----------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
body {background-color:#0E1117;}
h1,h2,h3 {color:white;}
</style>
""", unsafe_allow_html=True)

# ---------------- DOWNLOAD NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

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
    "neutral":"😐"
}

ALL_WORDS = sorted(set().union(*EMO_DICT.values()))

# ---------------- NLP ----------------

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

st.title("🧠 Emotion Detection System")
st.caption("Professional AI Emotion Analyzer")

text = st.text_area("Enter your text", height=120)

if st.button("Analyze Emotion"):

    if not text.strip():
        st.warning("Please type something")

    else:
        emotion, probs = predict(text)

        st.subheader(f"{EMO_EMOJI.get(emotion)} Detected Emotion: **{emotion.upper()}**")

        st.divider()

        st.markdown("### 📊 Confidence Levels")

        for emo,val in probs.items():
            st.write(f"{emo.capitalize()} — {int(val*100)}%")
            st.progress(val)

        st.success("Analysis complete ✔️")

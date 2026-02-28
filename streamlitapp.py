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

# Hinglish + Bengali phrases
MULTI_LANG_DICT = {
    "joy": {"accha lag raha","bhalo lagche","khushi","maja lagche"},
    "sadness": {"bhalo lagchena","accha nahi lag raha","dukhi","mon kharap"},
    "anger": {"gussa","rag","ragi"},
    "fear": {"dar lag raha","voy lagche","bhoy"}
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

# 🧚 Emotion Fairy
EMO_FAIRY = {
    "joy":"🧚‍♀️✨",
    "anger":"🧚‍♂️🔥",
    "sadness":"🧚‍♀️💙",
    "fear":"🧚‍♂️🌫",
    "disgust":"🧚‍♀️🧪",
    "surprise":"🧚‍♂️🌟"
}

ALL_EMOTIONS = list(EMO_DICT.keys())

# ---------------- NLP ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]

def detect_emotion(text):
    text_lower = text.lower()

    # Multi-language detection
    for emo, phrases in MULTI_LANG_DICT.items():
        for phrase in phrases:
            if phrase in text_lower:
                return emo, {emo:1.0}

    tokens = preprocess(text)
    scores = {e:0 for e in EMO_DICT}

    for tok in tokens:
        for emo,words in EMO_DICT.items():
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

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background:black;
    background-image:url("https://media.giphy.com/media/heOKY8nrJUMfK/giphy.gif");
    background-size:cover;
    background-position:center;
    background-repeat:no-repeat;
    background-attachment:fixed;
    color:white;
}

/* Bigger fonts */
.title {
    font-size:54px;
    text-align:center;
    font-weight:700;
}
.subtitle {
    font-size:22px;
    text-align:center;
    margin-bottom:20px;
}

/* Fairy header */
.fairy {
    font-size:70px;
    text-align:center;
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%{transform:translateY(0px);}
    50%{transform:translateY(-12px);}
    100%{transform:translateY(0px);}
}

/* Rotating Moon */
.moon {
    font-size:120px;
    text-align:center;
    animation: rotateMoon 30s linear infinite;
}
@keyframes rotateMoon {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Bars */
.bar-container {
    background:#1e1b4b;
    border-radius:20px;
    margin-bottom:10px;
}
.bar-fill {
    height:24px;
    border-radius:20px;
    transition: width 1.5s ease;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌌 Vibe Oracle</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak your vibe 🌙</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area("", height=160)

if st.button("🔮 Reveal the Vibe"):

    if not text.strip():
        st.warning("Say something… the stars are listening 👀")
    else:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        emotion, probs = detect_emotion(translated)

        color = EMO_COLOR[emotion]
        moon = EMO_MOON[emotion]
        fairy = EMO_FAIRY[emotion]

        # Show all emotions including 0%
        full_probs = {emo:0.0 for emo in ALL_EMOTIONS}
        for k,v in probs.items():
            full_probs[k]=v

        # Aura glow
        st.markdown(f"""
        <style>
        .block-container {{
            box-shadow:0 0 80px {color};
            border-radius:25px;
        }}
        </style>
        """, unsafe_allow_html=True)

        # 🧚 Fairy at Top
        st.markdown(f'<div class="fairy">{fairy}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="moon">{moon}</div>', unsafe_allow_html=True)
        st.markdown(f"## {EMO_EMOJI[emotion]} Dominant Vibe: **{emotion.upper()}**")

        st.markdown("### 🌌 Emotional Breakdown")

        for emo,val in full_probs.items():
            percent=int(val*100)

            if percent==100:
                bar_color="#00ffff"
            else:
                bar_color=EMO_COLOR[emo]

            st.markdown(f"""
            <div class="bar-container">
                <div class="bar-fill" style="width:{percent}%; background:{bar_color};"></div>
            </div>
            <small>{EMO_EMOJI[emo]} {emo.capitalize()} — {percent}%</small>
            """, unsafe_allow_html=True)

        st.success("✨ Cosmic vibe decoded by the fairy.")

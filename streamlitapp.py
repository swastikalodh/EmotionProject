import streamlit as st
import streamlit.components.v1 as components
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

EMO_COLOR = {
    "joy":"#facc15",
    "anger":"#ef4444",
    "sadness":"#60a5fa",
    "fear":"#a78bfa",
    "disgust":"#22c55e",
    "surprise":"#f472b6"
}

EMO_EMOJI = {
    "joy":"😊",
    "anger":"😡",
    "sadness":"😢",
    "fear":"😨",
    "disgust":"🤢",
    "surprise":"😲"
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

# ---------------- PARTICLE ENGINE ----------------
components.html("""
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<div id="particles-js"></div>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 80},
    "size": {"value": 2},
    "move": {"speed": 0.8},
    "line_linked": {"enable": true, "opacity": 0.2}
  }
});
</script>
<style>
#particles-js {
 position:fixed;
 width:100%;
 height:100%;
 top:0;
 left:0;
 z-index:-1;
}
</style>
""", height=0)

# ---------------- BASE STYLE ----------------
st.markdown("""
<style>
.stApp { background:black; color:white; }

.moon {
    font-size:90px;
    text-align:center;
    animation: rotateMoon 25s linear infinite;
}
@keyframes rotateMoon {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.bar-container {
    background:#222;
    border-radius:20px;
    margin-bottom:10px;
}
.bar {
    height:20px;
    border-radius:20px;
    animation: growBar 1.5s ease forwards;
}
@keyframes growBar {
    from { width:0%; }
}

svg { width:100%; height:120px; }
.star { fill:white; animation: twinkle 2s infinite alternate; }
@keyframes twinkle {
    from { opacity:0.4; }
    to { opacity:1; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌌 Vibe Oracle ✨")

# ---------------- FORM ----------------
with st.form("vibe_form"):
    user_text = st.text_area("Type your vibe (Press Enter)", height=150)
    submitted = st.form_submit_button("🔮 Reveal")

if submitted and user_text.strip():

    translated = GoogleTranslator(source='auto', target='en').translate(user_text)
    emotion, probs = detect_emotion(translated)

    color = EMO_COLOR[emotion]
    emoji = EMO_EMOJI[emotion]
    confidence = max(probs.values())

    # dynamic glow + moon brightness
    st.markdown(f"""
    <style>
    .block-container {{
        box-shadow:0 0 60px {color};
        border-radius:20px;
    }}
    .moon {{
        filter: brightness({0.5 + confidence});
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="moon">🌙</div>', unsafe_allow_html=True)
    st.markdown(f"## {emoji} Dominant Vibe: **{emotion.upper()}**")

    # animated constellation
    st.markdown("""
    <svg viewBox="0 0 200 100">
      <circle class="star" cx="30" cy="40" r="3"/>
      <circle class="star" cx="80" cy="20" r="3"/>
      <circle class="star" cx="130" cy="50" r="3"/>
      <circle class="star" cx="170" cy="30" r="3"/>
      <line x1="30" y1="40" x2="80" y2="20" stroke="white"/>
      <line x1="80" y1="20" x2="130" y2="50" stroke="white"/>
      <line x1="130" y1="50" x2="170" y2="30" stroke="white"/>
    </svg>
    """, unsafe_allow_html=True)

    st.markdown("### 🌌 Vibe Breakdown")

    for emo,val in probs.items():
        width=int(val*100)
        st.markdown(f"""
        <div class="bar-container">
            <div class="bar" style="width:{width}%; background:{color};"></div>
        </div>
        <small>{emo.capitalize()} — {width}%</small>
        """, unsafe_allow_html=True)

    st.success("✨ The cosmos has decoded your vibe.")

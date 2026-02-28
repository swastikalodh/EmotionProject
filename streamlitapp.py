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
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# ---------------- EMOTION DATA ----------------
EMO_DICT = {
    "joy":     {"happy","love","awesome","amazing","great","fantastic","smile"},
    "anger":   {"angry","hate","furious","mad","annoyed"},
    "sadness": {"sad","cry","lonely","depressed","heartbroken"},
    "fear":    {"scared","panic","terrified","afraid","nervous"},
    "disgust": {"gross","nasty","eww"},
    "surprise":{"wow","unexpected","omg","shocked"}
}

MULTI_LANG_DICT = {
    "joy":     {"accha lag raha","bhalo lagche","khushi","maja lagche"},
    "sadness": {"bhalo lagchena","accha nahi lag raha","dukhi","mon kharap"},
    "anger":   {"gussa","rag","ragi"},
    "fear":    {"dar lag raha","voy lagche","bhoy"}
}

EMO_EMOJI = {
    "joy":"😊","anger":"😡","sadness":"😢",
    "fear":"😨","disgust":"🤢","surprise":"😲"
}

EMO_COLOR = {
    "joy":     "#fde68a",
    "anger":   "#ff6b6b",
    "sadness": "#93c5fd",
    "fear":    "#c4b5fd",
    "disgust": "#6ee7b7",
    "surprise":"#f9a8d4"
}

EMO_GLOW = {
    "joy":     "#facc15",
    "anger":   "#ef4444",
    "sadness": "#3b82f6",
    "fear":    "#8b5cf6",
    "disgust": "#22c55e",
    "surprise":"#ec4899"
}

EMO_MOON = {
    "joy":"🌕","anger":"🌖","sadness":"🌑",
    "fear":"🌘","disgust":"🌒","surprise":"🌔"
}

EMO_FAIRY = {
    "joy":"🧚‍♀️✨","anger":"🧚‍♂️🔥","sadness":"🧚‍♀️💙",
    "fear":"🧚‍♂️🌫","disgust":"🧚‍♀️🧪","surprise":"🧚‍♂️🌟"
}

EMO_PARTICLE = {
    "joy":     ("⭐","✨","💛","🌟","☀️"),
    "anger":   ("🔥","💢","⚡","🌋","💥"),
    "sadness": ("💧","🌧","💙","🫧","❄️"),
    "fear":    ("🌫","👻","🕷","🌑","💫"),
    "disgust": ("🍃","🧪","💚","🌿","🫛"),
    "surprise":("🎉","🌸","💖","🎊","🌈")
}

EMO_MESSAGE = {
    "joy":     "The cosmos overflows with your radiant light ✨",
    "anger":   "A storm rages within you — let it pass like thunder 🌩",
    "sadness": "Even in darkness, stars still shine 🌌",
    "fear":    "The unknown holds mystery, not just shadow 🌫",
    "disgust": "Your spirit rejects what doesn't serve it 🌿",
    "surprise":"The universe just winked at you 😉"
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
    for emo, phrases in MULTI_LANG_DICT.items():
        for phrase in phrases:
            if phrase in text_lower:
                return emo, {emo: 1.0}
    tokens = preprocess(text)
    scores = {e: 0 for e in EMO_DICT}
    for tok in tokens:
        for emo, words in EMO_DICT.items():
            if tok in words:
                scores[emo] += 1
    non_zero = {k: v for k, v in scores.items() if v > 0}
    if len(non_zero) == 1:
        emo = list(non_zero.keys())[0]
        return emo, {emo: 1.0}
    if len(non_zero) > 1:
        total = sum(non_zero.values())
        probs = {k: round(v/total, 2) for k, v in non_zero.items()}
        best = max(probs, key=probs.get)
        return best, probs
    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0:
        return "joy", {"joy": 1.0}
    else:
        return "sadness", {"sadness": 1.0}

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Raleway:ital,wght@0,300;0,400;1,300&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --star-color: rgba(255,255,255,0.8);
    --nebula1: #0d0221;
    --nebula2: #0a0015;
    --nebula3: #050010;
}

/* ---- RESET & BODY ---- */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--nebula3) !important;
    color: white !important;
    font-family: 'Raleway', sans-serif !important;
}

/* ---- ANIMATED STARFIELD BACKGROUND ---- */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 30%, rgba(100,0,200,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 60% 80% at 80% 70%, rgba(0,80,200,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 60% 10%, rgba(200,0,100,0.10) 0%, transparent 70%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='800'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='800' height='800' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E"),
        linear-gradient(135deg, #0d0221 0%, #050010 50%, #0a0015 100%);
    z-index: -2;
    animation: nebulaPulse 12s ease-in-out infinite alternate;
}

@keyframes nebulaPulse {
    0%   { filter: hue-rotate(0deg) brightness(1); }
    50%  { filter: hue-rotate(20deg) brightness(1.08); }
    100% { filter: hue-rotate(-15deg) brightness(0.95); }
}

/* ---- TWINKLING STARS LAYER ---- */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 10% 15%, white 100%, transparent),
        radial-gradient(1.5px 1.5px at 25% 40%, rgba(255,255,255,0.9) 100%, transparent),
        radial-gradient(1px 1px at 40% 8%, white 100%, transparent),
        radial-gradient(2px 2px at 55% 60%, rgba(255,255,255,0.7) 100%, transparent),
        radial-gradient(1px 1px at 70% 25%, white 100%, transparent),
        radial-gradient(1.5px 1.5px at 85% 50%, rgba(255,255,255,0.8) 100%, transparent),
        radial-gradient(1px 1px at 92% 10%, white 100%, transparent),
        radial-gradient(2px 2px at 5% 80%, rgba(255,255,255,0.6) 100%, transparent),
        radial-gradient(1px 1px at 35% 90%, white 100%, transparent),
        radial-gradient(1.5px 1.5px at 60% 85%, rgba(255,255,255,0.9) 100%, transparent),
        radial-gradient(1px 1px at 78% 95%, white 100%, transparent),
        radial-gradient(2px 2px at 48% 50%, rgba(200,180,255,0.6) 100%, transparent),
        radial-gradient(1px 1px at 20% 65%, rgba(180,220,255,0.8) 100%, transparent);
    z-index: -1;
    animation: starTwinkle 6s ease-in-out infinite alternate;
}

@keyframes starTwinkle {
    0%   { opacity: 0.6; transform: scale(1); }
    50%  { opacity: 1.0; transform: scale(1.02); }
    100% { opacity: 0.7; transform: scale(0.98); }
}

/* ---- SHOOTING STAR ---- */
.stApp .block-container::before {
    content: '';
    position: fixed;
    top: 20%;
    left: -10%;
    width: 3px;
    height: 3px;
    background: white;
    border-radius: 50%;
    box-shadow: 0 0 6px 2px rgba(255,255,255,0.6), 80px 0 0 -1px rgba(255,255,255,0.3), 150px 0 0 -2px transparent;
    animation: shootingStar 8s linear infinite;
    z-index: 0;
}

@keyframes shootingStar {
    0%   { transform: translate(-100px, 0) rotate(-30deg);  opacity: 0; }
    5%   { opacity: 1; }
    30%  { transform: translate(110vw, 60px) rotate(-30deg); opacity: 0; }
    100% { transform: translate(110vw, 60px) rotate(-30deg); opacity: 0; }
}

/* ---- LAYOUT ---- */
.block-container {
    padding: 2rem 1.5rem 4rem !important;
    max-width: 780px !important;
    background: rgba(10,0,30,0.55) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(180,100,255,0.18) !important;
    border-radius: 32px !important;
    position: relative;
}

/* ---- TITLE ---- */
.oracle-title {
    font-family: 'Cinzel Decorative', cursive;
    font-size: clamp(36px, 8vw, 64px);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #e879f9, #818cf8, #38bdf8, #34d399, #fbbf24);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleShimmer 4s ease-in-out infinite, titleEntrance 1.2s cubic-bezier(0.22,1,0.36,1) both;
    text-shadow: none;
    letter-spacing: 2px;
    margin-bottom: 4px;
}
@keyframes titleShimmer {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes titleEntrance {
    from { opacity: 0; transform: translateY(-40px) scale(0.85); filter: blur(8px); }
    to   { opacity: 1; transform: translateY(0)   scale(1);    filter: blur(0); }
}

.oracle-subtitle {
    font-family: 'Raleway', sans-serif;
    font-style: italic;
    font-weight: 300;
    font-size: 18px;
    text-align: center;
    color: rgba(200,180,255,0.75);
    letter-spacing: 4px;
    text-transform: uppercase;
    animation: fadeUp 1.4s cubic-bezier(0.22,1,0.36,1) 0.3s both;
    margin-bottom: 32px;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- ORBITING ORBS ---- */
.orb-ring {
    position: relative;
    width: 160px;
    height: 160px;
    margin: 0 auto 16px;
}
.orb-center {
    position: absolute;
    inset: 20px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.3), transparent 60%),
                radial-gradient(circle, var(--emo-color, #818cf8), transparent 80%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 52px;
    animation: orbPulse 3s ease-in-out infinite;
    box-shadow: 0 0 40px var(--emo-glow, #818cf8), 0 0 80px rgba(0,0,0,0.6);
    z-index: 2;
}
@keyframes orbPulse {
    0%,100% { transform: scale(1);    box-shadow: 0 0 30px var(--emo-glow, #818cf8), 0 0 60px rgba(0,0,0,0.5); }
    50%      { transform: scale(1.08); box-shadow: 0 0 60px var(--emo-glow, #818cf8), 0 0 100px rgba(0,0,0,0.4); }
}
.orb-orbit {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 1.5px solid rgba(255,255,255,0.12);
    animation: orbitSpin 6s linear infinite;
}
.orb-orbit::before {
    content: '✦';
    position: absolute;
    top: -8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 14px;
    color: rgba(255,255,255,0.8);
}
@keyframes orbitSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.orb-orbit-2 {
    position: absolute;
    inset: -14px;
    border-radius: 50%;
    border: 1px dashed rgba(255,255,255,0.06);
    animation: orbitSpin 12s linear infinite reverse;
}
.orb-orbit-2::after {
    content: '◆';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 10px;
    color: rgba(200,150,255,0.6);
}

/* ---- FAIRY ---- */
.fairy {
    font-size: 62px;
    text-align: center;
    display: block;
    animation: fairyFloat 3s ease-in-out infinite, fairySpin 10s linear infinite;
    filter: drop-shadow(0 0 12px rgba(255,200,100,0.8));
    margin: 8px 0;
}
@keyframes fairyFloat {
    0%,100% { transform: translateY(0)   rotate(0deg);   }
    25%      { transform: translateY(-10px) rotate(5deg);  }
    75%      { transform: translateY(-5px)  rotate(-5deg); }
}
@keyframes fairySpin {
    0%,100% { filter: drop-shadow(0 0 12px rgba(255,200,100,0.8)) hue-rotate(0deg);   }
    50%      { filter: drop-shadow(0 0 20px rgba(255,200,100,1.0)) hue-rotate(60deg);  }
}

/* ---- PARTICLES ---- */
.particle-ring {
    position: relative;
    height: 60px;
    overflow: visible;
    text-align: center;
    margin: 8px 0;
}
.particle {
    position: absolute;
    font-size: 20px;
    animation: particleDrift var(--dur, 3s) ease-in-out infinite var(--delay, 0s);
    opacity: 0;
}
@keyframes particleDrift {
    0%   { transform: translate(0,0)      rotate(0deg)   scale(0);   opacity: 0; }
    15%  { opacity: 1; }
    50%  { transform: translate(var(--tx,20px), -50px) rotate(180deg) scale(1.2); opacity: 0.8; }
    85%  { opacity: 0.3; }
    100% { transform: translate(var(--tx2,40px), -90px) rotate(360deg) scale(0);  opacity: 0; }
}

/* ---- DOMINANT VIBE LABEL ---- */
.vibe-label {
    font-family: 'Cinzel Decorative', cursive;
    font-size: clamp(20px, 4vw, 30px);
    text-align: center;
    letter-spacing: 3px;
    margin: 16px 0 4px;
    animation: glowPulse 2s ease-in-out infinite;
}
@keyframes glowPulse {
    0%,100% { text-shadow: 0 0 10px var(--emo-glow, #818cf8), 0 0 30px rgba(0,0,0,0); }
    50%      { text-shadow: 0 0 20px var(--emo-glow, #818cf8), 0 0 50px var(--emo-glow, #818cf8); }
}

.vibe-message {
    font-family: 'Raleway', sans-serif;
    font-style: italic;
    font-weight: 300;
    font-size: 15px;
    text-align: center;
    color: rgba(220,210,255,0.75);
    letter-spacing: 1px;
    margin-bottom: 24px;
    animation: fadeUp 0.8s ease both;
}

/* ---- SECTION HEADER ---- */
.breakdown-header {
    font-family: 'Cinzel Decorative', cursive;
    font-size: 14px;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: rgba(180,160,255,0.7);
    text-align: center;
    margin: 24px 0 16px;
    position: relative;
}
.breakdown-header::before,
.breakdown-header::after {
    content: '✦';
    margin: 0 12px;
    opacity: 0.5;
}

/* ---- ANIMATED BARS ---- */
.bar-wrap {
    margin-bottom: 14px;
    animation: barEntrance 0.6s cubic-bezier(0.22,1,0.36,1) var(--bar-delay, 0s) both;
}
@keyframes barEntrance {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
.bar-track {
    background: rgba(30,20,60,0.8);
    border-radius: 100px;
    height: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    position: relative;
}
.bar-fill {
    height: 100%;
    border-radius: 100px;
    position: relative;
    animation: barGrow 1.6s cubic-bezier(0.22,1,0.36,1) var(--bar-delay, 0s) both;
    overflow: hidden;
}
@keyframes barGrow {
    from { width: 0 !important; }
    to   { width: var(--bar-w, 0%); }
}
.bar-fill::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.25) 50%, transparent 100%);
    animation: barShine 2s linear infinite;
}
@keyframes barShine {
    from { transform: translateX(-100%); }
    to   { transform: translateX(200%); }
}
.bar-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
    font-size: 12px;
    letter-spacing: 1px;
    color: rgba(200,190,240,0.7);
    font-family: 'Raleway', sans-serif;
    font-weight: 300;
}

/* ---- SUCCESS BANNER ---- */
.cosmic-banner {
    margin-top: 28px;
    padding: 14px 20px;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    font-family: 'Cinzel Decorative', cursive;
    font-size: 12px;
    letter-spacing: 3px;
    color: rgba(200,180,255,0.8);
    animation: bannerGlow 3s ease-in-out infinite;
}
@keyframes bannerGlow {
    0%,100% { border-color: rgba(255,255,255,0.08); }
    50%      { border-color: rgba(200,150,255,0.35); box-shadow: 0 0 20px rgba(150,100,255,0.15); }
}

/* ---- TEXTAREA ---- */
.stTextArea textarea {
    background: rgba(20,10,50,0.7) !important;
    border: 1px solid rgba(150,100,255,0.3) !important;
    border-radius: 16px !important;
    color: rgba(230,220,255,0.9) !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 16px !important;
    letter-spacing: 0.5px !important;
    resize: none !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(180,100,255,0.7) !important;
    box-shadow: 0 0 20px rgba(150,80,255,0.2) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: rgba(150,120,200,0.5) !important; }

/* ---- BUTTON ---- */
.stButton > button {
    width: 100%;
    padding: 16px 32px !important;
    background: linear-gradient(135deg, #6d28d9, #4f46e5, #7c3aed) !important;
    background-size: 200% 200% !important;
    border: none !important;
    border-radius: 100px !important;
    color: white !important;
    font-family: 'Cinzel Decorative', cursive !important;
    font-size: 15px !important;
    letter-spacing: 3px !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.3s !important;
    box-shadow: 0 0 30px rgba(109,40,217,0.5), 0 4px 20px rgba(0,0,0,0.4) !important;
    animation: btnShimmer 4s linear infinite !important;
    margin-top: 12px !important;
}
@keyframes btnShimmer {
    0%,100% { background-position: 0% 50%; box-shadow: 0 0 30px rgba(109,40,217,0.5), 0 4px 20px rgba(0,0,0,0.4); }
    50%      { background-position: 100% 50%; box-shadow: 0 0 50px rgba(109,40,217,0.8), 0 4px 30px rgba(0,0,0,0.5); }
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ---- HIDE STREAMLIT CHROME ---- */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.markdown('<h1 class="oracle-title">🌌 Vibe Oracle</h1>', unsafe_allow_html=True)
st.markdown('<p class="oracle-subtitle">Speak your vibe &nbsp;·&nbsp; decode the cosmos</p>', unsafe_allow_html=True)

# ---- INPUT ----
text = st.text_area("", height=160, placeholder="Pour your soul here… the stars are listening ✨")

if st.button("🔮 Reveal the Vibe"):

    if not text.strip():
        st.markdown("""
        <div style="text-align:center; padding:16px; border-radius:16px;
             background:rgba(255,100,100,0.08); border:1px solid rgba(255,100,100,0.2);
             color:rgba(255,180,180,0.9); font-style:italic; letter-spacing:1px;">
            🌌 Say something… the oracle awaits your truth
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("✨ Consulting the cosmos…"):
            translated = GoogleTranslator(source='auto', target='en').translate(text)
        emotion, probs = detect_emotion(translated)

        color = EMO_COLOR[emotion]
        glow  = EMO_GLOW[emotion]
        moon  = EMO_MOON[emotion]
        fairy = EMO_FAIRY[emotion]
        particles = EMO_PARTICLE[emotion]
        message = EMO_MESSAGE[emotion]

        full_probs = {emo: 0.0 for emo in ALL_EMOTIONS}
        for k, v in probs.items():
            full_probs[k] = v

        # Dynamic aura glow on container
        st.markdown(f"""
        <style>
        .block-container {{
            box-shadow: 0 0 100px {glow}40, 0 0 40px {glow}20 !important;
            border-color: {glow}44 !important;
        }}
        .vibe-label {{ --emo-glow: {glow}; color: {color}; }}
        .orb-center  {{ --emo-color: {color}; --emo-glow: {glow}; }}
        </style>
        """, unsafe_allow_html=True)

        # ---- FAIRY ----
        st.markdown(f'<span class="fairy">{fairy}</span>', unsafe_allow_html=True)

        # ---- PARTICLES ----
        particle_html = '<div class="particle-ring">'
        positions = [(-120,-80,-40,0,40,80,120)[i % 7] for i in range(5)]
        tx2s      = [(-160,-110,-60,20,60,110,160)[i % 7] for i in range(5)]
        for i, (p, tx, tx2) in enumerate(zip(particles, positions, tx2s)):
            dur = 2.5 + i * 0.4
            delay = i * 0.35
            particle_html += f"""
            <span class="particle" style="
                left:calc(50% + {tx}px);
                top:10px;
                --dur:{dur}s;
                --delay:{delay}s;
                --tx:{tx}px;
                --tx2:{tx2}px;">{p}</span>"""
        particle_html += '</div>'
        st.markdown(particle_html, unsafe_allow_html=True)

        # ---- ORBITING ORB ----
        st.markdown(f"""
        <div class="orb-ring">
            <div class="orb-orbit-2"></div>
            <div class="orb-orbit"></div>
            <div class="orb-center">{moon}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- VIBE LABEL ----
        st.markdown(f"""
        <div class="vibe-label">{EMO_EMOJI[emotion]} {emotion.upper()}</div>
        <div class="vibe-message">{message}</div>
        """, unsafe_allow_html=True)

        # ---- BREAKDOWN HEADER ----
        st.markdown('<div class="breakdown-header">Emotional Spectrum</div>', unsafe_allow_html=True)

        # ---- ANIMATED BARS ----
        for i, (emo, val) in enumerate(full_probs.items()):
            percent = int(val * 100)
            bar_color = EMO_GLOW[emo]
            delay = i * 0.12

            if percent == 100:
                bar_bg = f"linear-gradient(90deg, {bar_color}, #ffffff88, {bar_color})"
            elif percent > 0:
                bar_bg = f"linear-gradient(90deg, {bar_color}cc, {bar_color})"
            else:
                bar_bg = "rgba(80,60,120,0.3)"

            st.markdown(f"""
            <div class="bar-wrap" style="--bar-delay:{delay}s;">
                <div class="bar-track">
                    <div class="bar-fill" style="width:{percent}%; background:{bar_bg}; --bar-delay:{delay}s; --bar-w:{percent}%;"></div>
                </div>
                <div class="bar-label">
                    <span>{EMO_EMOJI[emo]} &nbsp;{emo.capitalize()}</span>
                    <span style="color:{bar_color}; font-weight:600;">{percent}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---- COSMIC BANNER ----
        st.markdown(f"""
        <div class="cosmic-banner">
            ✦ &nbsp; Vibe decoded by the Cosmic Fairy &nbsp; ✦
        </div>
        """, unsafe_allow_html=True)

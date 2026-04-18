"""
🌌 Vibe Oracle — Full Streamlit App
Detects emotional vibe of user input text with a mystical, visually rich UI.
"""

import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# ── Download required NLTK data ──────────────────────────────────────────────
for pkg in ["vader_lexicon", "stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ── Emotion Data Dictionaries ─────────────────────────────────────────────────

EMOTION_KEYWORDS = {
    "joy":      ["happy", "happiness", "joyful", "excited", "love", "wonderful", "amazing",
                 "great", "fantastic", "cheerful", "delighted", "thrilled", "bliss", "elated",
                 "ecstatic", "glad", "laugh", "smile", "celebrate", "fun", "enjoy", "grateful",
                 "awesome", "brilliant", "content", "pleased", "radiant", "euphoric"],
    "anger":    ["angry", "anger", "furious", "rage", "mad", "hate", "irritated", "annoyed",
                 "outraged", "frustrated", "enraged", "livid", "fuming", "hostile", "bitter",
                 "resentful", "aggressive", "violent", "disgusted", "infuriated", "explode"],
    "sadness":  ["sad", "unhappy", "depressed", "miserable", "heartbroken", "grief", "cry",
                 "sorrow", "mournful", "hopeless", "lonely", "gloomy", "melancholy", "despair",
                 "desolate", "tragic", "painful", "lost", "tears", "devastated", "anguish"],
    "fear":     ["afraid", "fear", "scared", "terrified", "anxious", "nervous", "panic",
                 "dread", "horror", "terror", "phobia", "worried", "uneasy", "apprehensive",
                 "trembling", "fright", "nightmare", "shock", "startled", "petrified"],
    "disgust":  ["disgusting", "gross", "revolting", "nasty", "awful", "yuck", "repulsed",
                 "sick", "vomit", "nauseating", "horrible", "repulsive", "filthy", "foul",
                 "unpleasant", "loathe", "abhorrent", "putrid", "hideous"],
    "surprise": ["surprised", "shocked", "astonished", "amazed", "unexpected", "wow",
                 "unbelievable", "incredible", "stunning", "remarkable", "astounded",
                 "speechless", "gasp", "omg", "whoa", "sudden", "startling", "jaw-dropping"],
}

MULTILANG_PHRASES = {
    # Hinglish
    "joy":      ["bahut maza", "kitna maza", "so happy", "bohot khushi", "maja aa gaya",
                 "full masti", "dil khush", "ek number", "bhai wah", "acha lag raha"],
    "anger":    ["bahut gussa", "bura lag raha", "kuch nahi chahiye", "bohot bura",
                 "chup raho", "teri toh", "faltu baat", "kya bakwas", "dimag mat kha"],
    "sadness":  ["bahut dukh", "rona aa raha", "dil toot gaya", "ek dum sad",
                 "kuch nahi ho raha", "akele hain", "bahut bura lag raha"],
    "fear":     ["bahut dar lag raha", "dar gaya", "dara hua", "bhoot jaisa",
                 "itna darna", "andhera"],
    "disgust":  ["chhi chhi", "yuck yaar", "kya bakwas hai", "bilkul pasand nahi",
                 "ganda hai", "ulti aa rahi"],
    "surprise": ["arre wah", "yaar kya baat", "sach mein", "aisa kaise", "oh my god yaar",
                 "kya hua", "kitni badi baat"],

    # Bengali phrases
    "joy":      ["অনেক মজা", "খুব ভালো", "আনন্দ", "হাসি", "দারুণ"],
    "anger":    ["রাগ হচ্ছে", "খুব রাগ", "বিরক্ত", "ঘেন্না"],
    "sadness":  ["কান্না পাচ্ছে", "খুব কষ্ট", "মন খারাপ", "দুঃখ"],
    "fear":     ["ভয় লাগছে", "ভয় পাচ্ছি", "ভয়ংকর"],
    "disgust":  ["ঘেন্না লাগছে", "বাজে"],
    "surprise": ["অবাক", "আশ্চর্য", "অবিশ্বাস্য"],
}

EMOTION_EMOJIS = {
    "joy":      "😄",
    "anger":    "😡",
    "sadness":  "😢",
    "fear":     "😱",
    "disgust":  "🤢",
    "surprise": "😲",
}

EMOTION_COLORS = {
    "joy":      "#FFD700",
    "anger":    "#FF4500",
    "sadness":  "#4169E1",
    "fear":     "#8A2BE2",
    "disgust":  "#228B22",
    "surprise": "#FF69B4",
}

MOON_PHASES = {
    "joy":      "🌕",
    "anger":    "🌑",
    "sadness":  "🌘",
    "fear":     "🌒",
    "disgust":  "🌓",
    "surprise": "🌙",
}

FAIRY_EMOJIS = {
    "joy":      "🧚‍♀️",
    "anger":    "🔥🧚",
    "sadness":  "🧚‍♂️",
    "fear":     "👻🧚",
    "disgust":  "🧚🍄",
    "surprise": "✨🧚‍♀️",
}

# ── NLP Utilities ─────────────────────────────────────────────────────────────

lemmatizer   = WordNetLemmatizer()
sia          = SentimentIntensityAnalyzer()

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()


def preprocess(text: str) -> list[str]:
    """Lowercase, strip punctuation, tokenize, remove stopwords, lemmatize."""
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens


def translate_to_english(text: str) -> str:
    """Translate any language to English using GoogleTranslator."""
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated if translated else text
    except Exception:
        return text


def detect_emotion(raw_text: str) -> dict[str, float]:
    """
    Detect emotion from raw_text.
    Priority: multi-lang phrases → keyword match → VADER fallback.
    Returns a probability distribution over all 6 emotions.
    """
    emotions      = list(EMOTION_KEYWORDS.keys())
    scores        = {e: 0 for e in emotions}
    raw_lower     = raw_text.lower()

    # 1. Multi-language phrase detection (check original text)
    for emotion, phrases in MULTILANG_PHRASES.items():
        for phrase in phrases:
            if phrase.lower() in raw_lower:
                scores[emotion] += 2  # phrase hit weighted higher

    # Translate to English for keyword + VADER
    translated = translate_to_english(raw_text)
    tokens     = preprocess(translated)

    # 2. Keyword matching on translated text
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in tokens or kw in translated.lower():
                scores[emotion] += 1

    total = sum(scores.values())

    if total > 0:
        return {e: round(scores[e] / total, 4) for e in emotions}

    # 3. VADER fallback
    compound = sia.polarity_scores(translated)["compound"]
    if compound >= 0:
        scores["joy"] = 1.0
    else:
        scores["sadness"] = 1.0
    return scores


# ── Streamlit Page Config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌌 Vibe Oracle",
    page_icon="🔮",
    layout="centered",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Raleway:wght@300;400;600&display=swap');

/* ── Reset & Background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #000010 !important;
    color: #f0e6ff !important;
    font-family: 'Raleway', sans-serif;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stMain"], section.main {
    background: transparent !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 30%, rgba(80,0,120,0.35) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 70%, rgba(0,50,120,0.3) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(10,0,40,0.9) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Stars */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 10% 15%, white, transparent),
        radial-gradient(1px 1px at 25% 40%, rgba(255,255,255,0.8), transparent),
        radial-gradient(1.5px 1.5px at 40% 10%, white, transparent),
        radial-gradient(1px 1px at 55% 60%, rgba(255,255,255,0.6), transparent),
        radial-gradient(1px 1px at 70% 25%, white, transparent),
        radial-gradient(2px 2px at 85% 45%, rgba(255,255,255,0.9), transparent),
        radial-gradient(1px 1px at 15% 75%, rgba(255,255,255,0.7), transparent),
        radial-gradient(1.5px 1.5px at 60% 85%, white, transparent),
        radial-gradient(1px 1px at 90% 80%, rgba(255,255,255,0.8), transparent),
        radial-gradient(1px 1px at 35% 55%, rgba(255,255,255,0.5), transparent);
    pointer-events: none;
    z-index: 0;
    animation: twinkle 4s ease-in-out infinite alternate;
}

@keyframes twinkle {
    0%   { opacity: 0.6; }
    100% { opacity: 1; }
}

/* ── Main content wrapper ── */
.main-wrapper {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 1rem 0 2rem;
}

/* ── Title ── */
.oracle-title {
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(2rem, 6vw, 3.8rem);
    font-weight: 900;
    background: linear-gradient(135deg, #c084fc, #818cf8, #38bdf8, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    letter-spacing: 0.04em;
    margin-bottom: 0.2rem;
    animation: titleGlow 3s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    0%   { filter: drop-shadow(0 0 8px rgba(192,132,252,0.6)); }
    100% { filter: drop-shadow(0 0 20px rgba(56,189,248,0.8)); }
}

/* ── Subtitle ── */
.oracle-subtitle {
    font-family: 'Raleway', sans-serif;
    font-size: 1.2rem;
    color: #c4b5fd;
    letter-spacing: 0.15em;
    margin-bottom: 1.5rem;
    font-weight: 300;
}

/* ── Floating Fairy ── */
.fairy-float {
    font-size: 2.8rem;
    display: block;
    animation: fairyFloat 2.5s ease-in-out infinite;
    margin-bottom: 0.5rem;
}

@keyframes fairyFloat {
    0%, 100% { transform: translateY(0px) rotate(-5deg); }
    50%       { transform: translateY(-14px) rotate(5deg); }
}

/* ── Rotating Moon ── */
.moon-spin {
    display: inline-block;
    font-size: 2rem;
    animation: moonSpin 8s linear infinite;
}

@keyframes moonSpin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* ── Glow Aura Card ── */
.aura-card {
    border-radius: 20px;
    padding: 2rem 1.5rem;
    margin: 1.5rem auto;
    max-width: 600px;
    position: relative;
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    transition: box-shadow 0.5s ease;
}

/* ── Emotion Name ── */
.emotion-name {
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(1.8rem, 5vw, 3rem);
    font-weight: 700;
    letter-spacing: 0.08em;
    margin: 0.4rem 0;
    animation: emotionPulse 2s ease-in-out infinite alternate;
}

@keyframes emotionPulse {
    0%   { filter: brightness(1); }
    100% { filter: brightness(1.3) drop-shadow(0 0 12px currentColor); }
}

.emotion-emoji-big {
    font-size: 3.5rem;
    display: block;
    margin: 0.3rem 0;
    animation: emojiBounce 1.2s ease-in-out infinite;
}

@keyframes emojiBounce {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.2); }
}

/* ── Bar Chart ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    font-family: 'Raleway', sans-serif;
}

.bar-label {
    width: 80px;
    text-align: right;
    font-size: 0.85rem;
    color: #e2d9f3;
    font-weight: 600;
    text-transform: capitalize;
}

.bar-track {
    flex: 1;
    height: 16px;
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
}

.bar-fill {
    height: 100%;
    border-radius: 8px;
    animation: barGrow 1.2s cubic-bezier(0.23, 1, 0.32, 1) forwards;
    transform-origin: left;
}

@keyframes barGrow {
    from { width: 0%; }
    to   { width: var(--bar-width); }
}

.bar-pct {
    width: 45px;
    font-size: 0.8rem;
    color: #c4b5fd;
    font-weight: 600;
}

/* ── Input area override ── */
textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #f0e6ff !important;
    border: 1px solid rgba(168,85,247,0.4) !important;
    border-radius: 12px !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 1rem !important;
}

textarea:focus {
    border-color: rgba(168,85,247,0.9) !important;
    box-shadow: 0 0 20px rgba(168,85,247,0.3) !important;
}

/* ── Button override ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.5rem !important;
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.5) !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.7) !important;
}

/* ── Warning ── */
.warn-msg {
    color: #fbbf24;
    font-family: 'Raleway', sans-serif;
    font-size: 1rem;
    padding: 0.8rem 1.2rem;
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    background: rgba(251,191,36,0.08);
    margin-top: 1rem;
}

/* ── Divider ── */
.mystic-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), transparent);
    margin: 1.5rem 0;
}

/* ── Section label ── */
.section-label {
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── App Layout ────────────────────────────────────────────────────────────────

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# Title + Subtitle
st.markdown('<p class="oracle-title">🌌 Vibe Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="oracle-subtitle">Speak your vibe 🌙</p>', unsafe_allow_html=True)

# Input
user_input = st.text_area(
    label="",
    placeholder="Type anything… in any language 🔮",
    height=130,
    key="vibe_input",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 1.5, 2])
with col2:
    reveal = st.button("🔮 Reveal the Vibe", use_container_width=True)

if reveal:
    if not user_input.strip():
        st.markdown('<div class="warn-msg">⚠️ Please enter some text to reveal your vibe…</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("✨ Reading the cosmic vibrations…"):
            scores = detect_emotion(user_input)

        # Dominant emotion
        dominant  = max(scores, key=scores.get)
        color     = EMOTION_COLORS[dominant]
        emoji     = EMOTION_EMOJIS[dominant]
        fairy     = FAIRY_EMOJIS[dominant]
        moon      = MOON_PHASES[dominant]
        dominant_pct = int(scores[dominant] * 100)

        # ── Aura card ──
        glow = f"0 0 60px {color}55, 0 0 120px {color}22, inset 0 0 40px {color}11"
        st.markdown(f"""
        <div class="aura-card" style="box-shadow: {glow}; border-color: {color}44;">
            <span class="fairy-float">{fairy}</span>
            <span class="moon-spin">{moon}</span>
            <p class="section-label">Your Dominant Vibe</p>
            <span class="emotion-emoji-big">{emoji}</span>
            <p class="emotion-name" style="color:{color};">{dominant.upper()}</p>
            <p style="color:#c4b5fd; font-size:0.95rem; margin-top:0.2rem; font-family:'Raleway',sans-serif;">
                {fairy} &nbsp; {moon} &nbsp; {emoji}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Emotion Breakdown ──
        st.markdown('<hr class="mystic-divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label" style="text-align:center;">✦ Emotion Breakdown ✦</p>',
                    unsafe_allow_html=True)

        for emotion in EMOTION_KEYWORDS:
            pct   = int(scores[emotion] * 100)
            ec    = EMOTION_COLORS[emotion]
            ee    = EMOTION_EMOJIS[emotion]
            # Highlight full dominance
            fill_color = f"linear-gradient(90deg, {ec}, #fff5)" if pct == 100 else \
                         f"linear-gradient(90deg, {ec}cc, {ec}55)"
            bar_bg = f"rgba(255,255,255,0.05)"

            st.markdown(f"""
            <div class="bar-row">
                <span class="bar-label">{ee} {emotion}</span>
                <div class="bar-track" style="background:{bar_bg};">
                    <div class="bar-fill"
                         style="--bar-width:{pct}%;
                                width:{pct}%;
                                background:{fill_color};
                                box-shadow: 0 0 8px {ec}88;">
                    </div>
                </div>
                <span class="bar-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrapper

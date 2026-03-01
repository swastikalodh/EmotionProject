import streamlit as st
import re
import math

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="🌌 Vibe Oracle",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Lazy NLP imports (graceful degradation if not installed) ─────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer

    for pkg in ["stopwords", "wordnet", "vader_lexicon", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}" if pkg in ("stopwords","wordnet") else f"sentiment/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words("english"))
    _sia = SentimentIntensityAnalyzer()
    NLP_OK = True
except Exception:
    NLP_OK = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except Exception:
    TRANSLATOR_OK = False


# ── Emotion Definitions ───────────────────────────────────────────────────────
EMOTIONS = {
    "joy": {
        "emoji": "✨", "color": "#FFD700", "glow": "#FFD70088",
        "moon": "🌕", "fairy": "🧚‍♀️", "particles": ["⭐", "✨", "💫", "🌟", "🌙"],
        "message": "The cosmos sings with golden light through you. Your radiance bends the very fabric of the universe.",
    },
    "anger": {
        "emoji": "🔥", "color": "#FF4500", "glow": "#FF450088",
        "moon": "🌑", "fairy": "🧚‍♂️", "particles": ["🔥", "⚡", "💢", "🌋", "☄️"],
        "message": "The fire within you forges stars. Channel this cosmic fury into transformation.",
    },
    "sadness": {
        "emoji": "🌊", "color": "#4169E1", "glow": "#4169E188",
        "moon": "🌒", "fairy": "🧚", "particles": ["💧", "🌊", "🌧️", "❄️", "🌫️"],
        "message": "Even the moon weeps silver tears. Your depth of feeling is the universe knowing itself.",
    },
    "fear": {
        "emoji": "🌑", "color": "#800080", "glow": "#80008088",
        "moon": "🌘", "fairy": "🧝", "particles": ["🌑", "👁️", "🌀", "💜", "🕯️"],
        "message": "The unknown is where all stars are born. Breathe—the cosmos holds you in its infinite arms.",
    },
    "disgust": {
        "emoji": "🍃", "color": "#228B22", "glow": "#228B2288",
        "moon": "🌓", "fairy": "🧜", "particles": ["🍃", "🌿", "💚", "🌱", "🦋"],
        "message": "Your discernment is sacred. The universe honors those who know their truth.",
    },
    "surprise": {
        "emoji": "🌠", "color": "#FF69B4", "glow": "#FF69B488",
        "moon": "🌟", "fairy": "🧞", "particles": ["🌠", "💥", "🎆", "🎇", "✨"],
        "message": "The cosmos loves to astonish. You stand at the threshold of infinite possibility!",
    },
}

# ── Keyword Dictionaries ──────────────────────────────────────────────────────
MULTILINGUAL_PHRASES = {
    # Hindi
    "खुश": "joy", "खुशी": "joy", "प्यार": "joy", "आनंद": "joy",
    "गुस्सा": "anger", "क्रोध": "anger", "नफ़रत": "anger",
    "दुख": "sadness", "उदास": "sadness", "रोना": "sadness",
    "डर": "fear", "भय": "fear", "घबराहट": "fear",
    "घृणा": "disgust", "नापसंद": "disgust",
    "अचरज": "surprise", "हैरान": "surprise",
    # Bengali
    "আনন্দ": "joy", "খুশি": "joy", "ভালোবাসা": "joy",
    "রাগ": "anger", "ক্রোধ": "anger",
    "দুঃখ": "sadness", "কান্না": "sadness",
    "ভয়": "fear", "আতঙ্ক": "fear",
    "ঘৃণা": "disgust",
    "আশ্চর্য": "surprise", "বিস্মিত": "surprise",
}

ENGLISH_KEYWORDS = {
    "joy": ["happy","happiness","joy","joyful","love","loving","elated","cheerful","excited","wonderful","fantastic","amazing","great","awesome","bliss","blissful","delight","delighted","thrilled","ecstatic","glad","pleased","content","laugh","laughing","smile","smiling","celebrate","celebration","fun","enjoy","enjoying","grateful","gratitude","euphoric","jubilant","radiant","gleeful","overjoyed"],
    "anger": ["angry","anger","furious","rage","mad","irritated","annoyed","frustrated","hate","hatred","outraged","livid","enraged","hostile","bitter","resentful","agitated","infuriated","disgusted","irate","wrathful","seething","fuming"],
    "sadness": ["sad","sadness","unhappy","depressed","depression","grief","sorrow","cry","crying","tears","lonely","loneliness","heartbroken","miserable","gloomy","melancholy","hopeless","devastated","despair","mourning","wretched","forlorn","somber","sorrowful","distressed","broken","lost","empty","numb"],
    "fear": ["afraid","fear","scared","frightened","terrified","anxious","anxiety","panic","worried","nervous","dread","phobia","horror","terror","uneasy","apprehensive","paranoid","timid","trembling","horrified","petrified","shaken","alarmed"],
    "disgust": ["disgusted","disgust","gross","revolting","repulsed","nauseated","sick","yuck","eww","nasty","vile","repel","repellent","abhorrent","loathe","loathing","offensive","foul","putrid","repugnant","despise"],
    "surprise": ["surprised","surprise","shocked","astonished","amazed","stunned","startled","unexpected","unbelievable","wow","whoa","incredible","mind-blowing","astounded","dumbfounded","speechless","bewildered","flabbergasted","dazzled"],
}


# ── NLP Preprocessing ─────────────────────────────────────────────────────────
def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    if NLP_OK:
        tokens = [t for t in tokens if t not in _stop_words]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def detect_emotion(raw_text: str) -> dict[str, float]:
    # 1. Multilingual phrase check
    scores = {e: 0 for e in EMOTIONS}
    for phrase, emotion in MULTILINGUAL_PHRASES.items():
        if phrase in raw_text:
            scores[emotion] += 2

    # 2. English keyword matching
    tokens = preprocess(raw_text)
    for emotion, keywords in ENGLISH_KEYWORDS.items():
        for token in tokens:
            if token in keywords:
                scores[emotion] += 1

    total = sum(scores.values())

    # 3. VADER fallback
    if total == 0:
        if NLP_OK:
            vs = _sia.polarity_scores(raw_text)
            compound = vs["compound"]
        else:
            # Simple fallback without VADER
            pos_words = sum(1 for t in tokens if t in ENGLISH_KEYWORDS["joy"] + ENGLISH_KEYWORDS["surprise"])
            neg_words = sum(1 for t in tokens if t in ENGLISH_KEYWORDS["sadness"] + ENGLISH_KEYWORDS["anger"] + ENGLISH_KEYWORDS["fear"])
            compound = (pos_words - neg_words) / max(len(tokens), 1)

        if compound >= 0.05:
            scores["joy"] = 1
        elif compound <= -0.05:
            scores["sadness"] = 1
        else:
            scores["surprise"] = 0.5
            scores["joy"] = 0.5
        total = sum(scores.values())

    if total == 0:
        scores["surprise"] = 1
        total = 1

    return {e: round(v / total * 100, 1) for e, v in scores.items()}


def dominant_emotion(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


# ── CSS Injection ─────────────────────────────────────────────────────────────
def inject_css(glow_color: str = "#b04aff88"):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&display=swap');

/* ── Reset & Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stDeployButton"],
.stDeployButton {{ display: none !important; }}
[data-testid="stAppViewContainer"] > .main {{ padding: 0 !important; }}

/* ── Cosmic Background ── */
body, [data-testid="stAppViewContainer"] {{
    background: #000005 !important;
    min-height: 100vh;
    overflow-x: hidden;
}}

[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, #1a0033 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, #001a33 0%, transparent 60%),
        radial-gradient(ellipse 100% 100% at 50% 50%, #0d001a 0%, #000005 100%);
    animation: nebulaShift 12s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}}

@keyframes nebulaShift {{
    0%  {{ filter: hue-rotate(0deg) brightness(1); }}
    50% {{ filter: hue-rotate(20deg) brightness(1.08); }}
    100%{{ filter: hue-rotate(-10deg) brightness(0.95); }}
}}

/* Fractal noise overlay */
[data-testid="stAppViewContainer"]::after {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    opacity: 0.35;
    pointer-events: none;
    z-index: 1;
}}

/* Twinkling stars */
.stars-layer {{
    position: fixed;
    inset: 0;
    z-index: 2;
    pointer-events: none;
    background-image:
        radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 40%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 40% 8%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 60%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 25%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(2px 2px at 80% 75%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 45%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 15% 80%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 35% 65%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 90%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 5%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 5% 55%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(2px 2px at 50% 30%, rgba(200,180,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 88% 18%, rgba(180,200,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 22% 95%, rgba(255,200,180,0.5) 0%, transparent 100%);
    animation: twinkle 4s ease-in-out infinite alternate;
}}
@keyframes twinkle {{
    0%  {{ opacity: 0.6; }}
    100%{{ opacity: 1; }}
}}

/* Shooting star */
.shooting-star {{
    position: fixed;
    top: 15%;
    left: -10%;
    width: 120px;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.9), transparent);
    border-radius: 50%;
    animation: shoot 6s linear infinite;
    z-index: 3;
    pointer-events: none;
}}
@keyframes shoot {{
    0%   {{ transform: translateX(0) translateY(0) rotate(25deg); opacity: 0; }}
    5%   {{ opacity: 1; }}
    40%  {{ opacity: 0.8; }}
    60%  {{ opacity: 0; transform: translateX(120vw) translateY(40vh) rotate(25deg); }}
    100% {{ transform: translateX(120vw) translateY(40vh) rotate(25deg); opacity: 0; }}
}}

/* ── Main container ── */
.main-container {{
    position: relative;
    z-index: 10;
    max-width: 760px;
    margin: 40px auto;
    padding: 48px 44px;
    background: rgba(10, 0, 30, 0.55);
    backdrop-filter: blur(24px) saturate(1.5);
    -webkit-backdrop-filter: blur(24px) saturate(1.5);
    border: 1px solid rgba(180, 100, 255, 0.25);
    border-radius: 28px;
    box-shadow:
        0 0 60px {glow_color},
        0 0 120px rgba(80, 0, 160, 0.2),
        inset 0 1px 0 rgba(255,255,255,0.1);
    animation: containerPulse 5s ease-in-out infinite alternate;
    transition: box-shadow 1s ease;
}}
@keyframes containerPulse {{
    0%  {{ box-shadow: 0 0 60px {glow_color}, 0 0 120px rgba(80,0,160,0.2), inset 0 1px 0 rgba(255,255,255,0.1); }}
    100%{{ box-shadow: 0 0 90px {glow_color}, 0 0 180px rgba(80,0,160,0.35), inset 0 1px 0 rgba(255,255,255,0.15); }}
}}

/* ── Typography ── */
.oracle-title {{
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #e8c0ff 0%, #b04aff 30%, #6af0ff 60%, #ffb3e6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 300% 300%;
    animation: gradientFlow 4s ease infinite;
    margin-bottom: 8px;
    letter-spacing: 0.04em;
    line-height: 1.2;
}}
@keyframes gradientFlow {{
    0%  {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100%{{ background-position: 0% 50%; }}
}}

.oracle-subtitle {{
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: rgba(200, 170, 255, 0.8);
    text-align: center;
    margin-bottom: 36px;
    letter-spacing: 0.12em;
    animation: fadeUp 1s ease 0.3s both;
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {{
    background: rgba(20, 0, 50, 0.7) !important;
    border: 1px solid rgba(180, 100, 255, 0.35) !important;
    border-radius: 16px !important;
    color: #e8d0ff !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    padding: 16px 20px !important;
    transition: border-color 0.4s, box-shadow 0.4s !important;
    resize: vertical !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border-color: rgba(180, 100, 255, 0.8) !important;
    box-shadow: 0 0 24px rgba(176, 74, 255, 0.4), 0 0 48px rgba(100, 0, 200, 0.2) !important;
    outline: none !important;
}}
[data-testid="stTextArea"] textarea::placeholder {{
    color: rgba(180, 140, 255, 0.45) !important;
}}
[data-testid="stTextArea"] label {{
    color: rgba(200, 170, 255, 0.7) !important;
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
}}

/* ── Button ── */
[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, #6a00ff, #b04aff, #ff69b4, #6a00ff) !important;
    background-size: 300% 300% !important;
    animation: btnShimmer 3s ease infinite !important;
    border: none !important;
    border-radius: 50px !important;
    color: #fff !important;
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    padding: 14px 40px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(176, 74, 255, 0.5) !important;
}}
[data-testid="stButton"] > button:hover {{
    transform: scale(1.04) !important;
    box-shadow: 0 8px 40px rgba(176, 74, 255, 0.7) !important;
}}
@keyframes btnShimmer {{
    0%  {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100%{{ background-position: 0% 50%; }}
}}

/* ── Warning banner ── */
.warn-banner {{
    background: rgba(255, 100, 0, 0.12);
    border: 1px solid rgba(255, 150, 50, 0.45);
    border-radius: 14px;
    padding: 14px 20px;
    color: #ffbb77;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    text-align: center;
    margin: 12px 0;
    letter-spacing: 0.06em;
}}

/* ── Emotion display ── */
.emotion-card {{
    text-align: center;
    padding: 32px 20px;
    animation: emotionReveal 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}}
@keyframes emotionReveal {{
    from {{ opacity: 0; transform: scale(0.8) translateY(20px); }}
    to   {{ opacity: 1; transform: scale(1) translateY(0); }}
}}

.emotion-emoji-big {{
    font-size: 3.5rem;
    display: block;
    animation: floatFairy 3s ease-in-out infinite;
    filter: drop-shadow(0 0 20px currentColor);
}}
@keyframes floatFairy {{
    0%, 100% {{ transform: translateY(0) rotate(-5deg); }}
    50%       {{ transform: translateY(-12px) rotate(5deg); }}
}}

.emotion-label {{
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(1.4rem, 3vw, 2.2rem);
    font-weight: 900;
    letter-spacing: 0.15em;
    margin: 12px 0 6px;
    animation: glowPulse 2s ease-in-out infinite;
}}
@keyframes glowPulse {{
    0%, 100% {{ text-shadow: 0 0 20px currentColor, 0 0 40px currentColor; }}
    50%       {{ text-shadow: 0 0 40px currentColor, 0 0 80px currentColor, 0 0 120px currentColor; }}
}}

.emotion-message {{
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: rgba(220, 200, 255, 0.85);
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.7;
    letter-spacing: 0.04em;
}}

/* ── Particles ── */
.particles-wrap {{
    display: flex;
    justify-content: center;
    gap: 16px;
    margin: 16px 0;
    flex-wrap: wrap;
}}
.particle {{
    font-size: 1.6rem;
    animation: particleFloat 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(255,255,255,0.5));
}}
.particle:nth-child(1) {{ animation-delay: 0s;   animation-duration: 2.2s; }}
.particle:nth-child(2) {{ animation-delay: 0.4s; animation-duration: 2.8s; }}
.particle:nth-child(3) {{ animation-delay: 0.8s; animation-duration: 2.4s; }}
.particle:nth-child(4) {{ animation-delay: 1.2s; animation-duration: 3.0s; }}
.particle:nth-child(5) {{ animation-delay: 1.6s; animation-duration: 2.6s; }}
@keyframes particleFloat {{
    0%, 100% {{ transform: translateY(0) scale(1); opacity: 0.8; }}
    50%       {{ transform: translateY(-18px) scale(1.2); opacity: 1; }}
}}

/* ── Orb system ── */
.orb-system {{
    position: relative;
    width: 120px;
    height: 120px;
    margin: 20px auto;
}}
.orb-center {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 2.4rem;
    z-index: 3;
    filter: drop-shadow(0 0 16px rgba(255,255,255,0.8));
    animation: orbPulse 2s ease-in-out infinite;
}}
@keyframes orbPulse {{
    0%, 100% {{ transform: translate(-50%, -50%) scale(1); filter: drop-shadow(0 0 16px rgba(255,255,255,0.8)); }}
    50%       {{ transform: translate(-50%, -50%) scale(1.15); filter: drop-shadow(0 0 32px rgba(255,255,255,1)); }}
}}
.orbit-ring {{
    position: absolute;
    inset: 0;
    border: 2px solid rgba(180, 100, 255, 0.5);
    border-radius: 50%;
    animation: orbitSpin 4s linear infinite;
}}
.orbit-ring::after {{
    content: '⚬';
    position: absolute;
    top: -8px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(180, 100, 255, 0.9);
    font-size: 1rem;
}}
.orbit-ring-2 {{
    position: absolute;
    inset: 8px;
    border: 1px dashed rgba(100, 200, 255, 0.3);
    border-radius: 50%;
    animation: orbitSpin 6s linear infinite reverse;
}}
@keyframes orbitSpin {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
}}

/* ── Spectrum bars ── */
.spectrum-title {{
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.75rem;
    color: rgba(180, 140, 255, 0.7);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    text-align: center;
    margin: 28px 0 18px;
}}
.bar-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    animation: barEntrance 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}}
.bar-row:nth-child(1) {{ animation-delay: 0.05s; }}
.bar-row:nth-child(2) {{ animation-delay: 0.15s; }}
.bar-row:nth-child(3) {{ animation-delay: 0.25s; }}
.bar-row:nth-child(4) {{ animation-delay: 0.35s; }}
.bar-row:nth-child(5) {{ animation-delay: 0.45s; }}
.bar-row:nth-child(6) {{ animation-delay: 0.55s; }}
@keyframes barEntrance {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
.bar-label {{
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    min-width: 68px;
    text-align: right;
}}
.bar-track {{
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}}
.bar-fill {{
    height: 100%;
    border-radius: 10px;
    position: relative;
    transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    overflow: hidden;
}}
.bar-fill::after {{
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: barShine 2s ease-in-out infinite;
}}
@keyframes barShine {{
    0%  {{ left: -100%; }}
    100%{{ left: 200%; }}
}}
.bar-pct {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.8rem;
    min-width: 38px;
    color: rgba(200, 170, 255, 0.7);
}}

/* ── Footer banner ── */
.cosmic-footer {{
    margin-top: 36px;
    padding: 16px 24px;
    border: 1px solid rgba(180, 100, 255, 0.35);
    border-radius: 14px;
    text-align: center;
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: rgba(200, 170, 255, 0.7);
    background: rgba(100, 0, 200, 0.08);
    animation: footerGlow 3s ease-in-out infinite alternate;
}}
@keyframes footerGlow {{
    0%  {{ box-shadow: 0 0 12px rgba(176,74,255,0.2); }}
    100%{{ box-shadow: 0 0 28px rgba(176,74,255,0.5), 0 0 60px rgba(100,0,200,0.2); }}
}}

/* ── Divider ── */
.cosmic-divider {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,100,255,0.4), transparent);
    margin: 24px 0;
}}

/* ── Spinner override ── */
[data-testid="stSpinner"] p {{
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    color: rgba(200, 170, 255, 0.85) !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.08em !important;
}}
</style>

<!-- Background layers -->
<div class="stars-layer"></div>
<div class="shooting-star"></div>
""", unsafe_allow_html=True)


# ── Render helpers ────────────────────────────────────────────────────────────
def render_title():
    st.markdown("""
<div class="oracle-title">🌌 Vibe Oracle</div>
<div class="oracle-subtitle">The cosmos whispers your truth back to you</div>
""", unsafe_allow_html=True)


def render_orb(moon_icon: str):
    st.markdown(f"""
<div class="orb-system">
    <div class="orbit-ring"></div>
    <div class="orbit-ring-2"></div>
    <div class="orb-center">{moon_icon}</div>
</div>
""", unsafe_allow_html=True)


def render_particles(particles: list[str]):
    items = "".join(f'<span class="particle">{p}</span>' for p in particles)
    st.markdown(f'<div class="particles-wrap">{items}</div>', unsafe_allow_html=True)


def render_emotion_card(emotion: str, probs: dict[str, float]):
    ed = EMOTIONS[emotion]
    color = ed["color"]
    pct = probs[emotion]

    st.markdown(f"""
<div class="emotion-card">
    <span class="emotion-emoji-big" style="color:{color}">{ed["fairy"]}</span>
    <div class="emotion-label" style="color:{color}">{emotion.upper()}</div>
    <div class="emotion-message">{ed["message"]}</div>
</div>
""", unsafe_allow_html=True)


def render_spectrum(probs: dict[str, float]):
    st.markdown('<div class="spectrum-title">✦ Emotional Spectrum ✦</div>', unsafe_allow_html=True)

    sorted_emotions = sorted(probs.items(), key=lambda x: -x[1])
    bars_html = ""
    for emotion, pct in sorted_emotions:
        ed = EMOTIONS[emotion]
        color = ed["color"]
        width = max(pct, 1)
        bars_html += f"""
<div class="bar-row">
    <div class="bar-label" style="color:{color}">{ed['emoji']} {emotion.capitalize()}</div>
    <div class="bar-track">
        <div class="bar-fill" style="width:{width}%; background: linear-gradient(90deg, {color}88, {color})"></div>
    </div>
    <div class="bar-pct">{pct:.0f}%</div>
</div>"""

    st.markdown(bars_html, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
<hr class="cosmic-divider"/>
<div class="cosmic-footer">✦ Vibe decoded by the Cosmic Fairy ✦</div>
""", unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Default glow before detection
    glow = st.session_state.get("glow_color", "#b04aff88")
    inject_css(glow_color=glow)

    # Main glassmorphic wrapper
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    render_title()

    st.markdown('<hr class="cosmic-divider"/>', unsafe_allow_html=True)

    user_text = st.text_area(
        "",
        placeholder="Pour your soul here… the stars are listening ✨",
        height=140,
        key="soul_input",
        label_visibility="collapsed",
    )

    submitted = st.button("🔮 Reveal the Vibe")

    if submitted:
        if not user_text.strip():
            st.markdown("""
<div class="warn-banner">
    🌙 The cosmos received only silence… speak your truth so the stars may answer.
</div>
""", unsafe_allow_html=True)
        else:
            with st.spinner("✨ Consulting the cosmos…"):
                probs = detect_emotion(user_text)

            dom = dominant_emotion(probs)
            ed = EMOTIONS[dom]

            # Update glow color for container (next render cycle)
            st.session_state["glow_color"] = ed["glow"]

            st.markdown('<hr class="cosmic-divider"/>', unsafe_allow_html=True)

            render_orb(ed["moon"])
            render_particles(ed["particles"])
            render_emotion_card(dom, probs)

            st.markdown('<hr class="cosmic-divider"/>', unsafe_allow_html=True)

            render_spectrum(probs)

    render_footer()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

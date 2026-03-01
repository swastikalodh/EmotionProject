"""
🌌 Vibe Oracle v2 — ULTRA MAGICAL COSMIC EDITION
Requirements: pip install streamlit nltk deep-translator
Then run: streamlit run vibe_oracle.py
"""

import streamlit as st
import re

st.set_page_config(
    page_title="🌌 Vibe Oracle",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── NLP imports ───────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    for pkg, path in [("stopwords","corpora/stopwords"),("wordnet","corpora/wordnet"),
                      ("vader_lexicon","sentiment/vader_lexicon"),("punkt","tokenizers/punkt")]:
        try: nltk.data.find(path)
        except LookupError: nltk.download(pkg, quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _stop_words  = set(stopwords.words("english"))
    _sia         = SentimentIntensityAnalyzer()
    NLP_OK = True
except Exception:
    NLP_OK = False

# ── Hex helper (defined early) ─────────────────────────────────────────────────
def rgb(h):
    h = h.strip().lstrip('#')
    if len(h) == 3: h = h[0]*2 + h[1]*2 + h[2]*2
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

# ── Emotion Data ──────────────────────────────────────────────────────────────
EMOTIONS = {
    "joy": {
        "emoji":"✨","color":"#FFD700","color2":"#FF8C00","glow":"#FFD700",
        "moon":"🌕","fairy":"🧚‍♀️","crystal":"🟡","aura":"Golden Solar Aura","rune":"ᚷ",
        "bg_from":"#1a1000","bg_to":"#0d0800",
        "particles":["⭐","✨","💫","🌟","☀️","🌈","💛","🎊","🎉","🌻"],
        "message":"The cosmos SINGS through you! Your radiance bends starlight itself. Even black holes lean closer to bask in your glow ✨",
    },
    "anger": {
        "emoji":"🔥","color":"#FF4500","color2":"#FF0000","glow":"#FF4500",
        "moon":"🌑","fairy":"🐉","crystal":"🔴","aura":"Crimson Dragon Aura","rune":"ᚦ",
        "bg_from":"#1a0000","bg_to":"#0d0000",
        "particles":["🔥","⚡","💢","🌋","☄️","💥","🔴","⚔️","🌪️","💣"],
        "message":"VOLCANIC POWER courses through your veins! The universe trembles at your passion. Forge galaxies from this sacred fire 🔥",
    },
    "sadness": {
        "emoji":"🌊","color":"#4169E1","color2":"#00BFFF","glow":"#4169E1",
        "moon":"🌒","fairy":"🧜‍♀️","crystal":"🔵","aura":"Midnight Ocean Aura","rune":"ᛚ",
        "bg_from":"#000d2a","bg_to":"#00061a",
        "particles":["💧","🌊","🌧️","❄️","🌫️","💙","🫧","🌀","🩵","🌙"],
        "message":"Even the ocean weeps silver tears, and it covers 71% of the Earth. Your depth of feeling IS the universe knowing itself 🌊",
    },
    "fear": {
        "emoji":"🌑","color":"#9400D3","color2":"#FF00FF","glow":"#9400D3",
        "moon":"🌘","fairy":"🦇","crystal":"🟣","aura":"Shadow Mystic Aura","rune":"ᚾ",
        "bg_from":"#130020","bg_to":"#0a0015",
        "particles":["👁️","🌀","💜","🕯️","🌑","🕸️","🦋","⚗️","🔮","🌙"],
        "message":"The void stares back — and it BLINKS first. Every star was born from darkness. You are becoming something magnificent 🔮",
    },
    "disgust": {
        "emoji":"🍃","color":"#00FF88","color2":"#228B22","glow":"#00FF88",
        "moon":"🌓","fairy":"🧝‍♀️","crystal":"🟢","aura":"Ancient Forest Aura","rune":"ᛃ",
        "bg_from":"#001500","bg_to":"#000d00",
        "particles":["🍃","🌿","💚","🌱","🦋","🌺","🍀","🌳","✨","🐍"],
        "message":"Your sacred NO is as powerful as any YES. The cosmos bows to those who know their own truth. You are untameable 🌿",
    },
    "surprise": {
        "emoji":"🌠","color":"#FF69B4","color2":"#FF1493","glow":"#FF69B4",
        "moon":"🌟","fairy":"🧞‍♀️","crystal":"🩷","aura":"Stardust Supernova Aura","rune":"ᛇ",
        "bg_from":"#200015","bg_to":"#14000e",
        "particles":["🌠","💥","🎆","🎇","✨","🌸","🎊","💖","⚡","🌈"],
        "message":"THE UNIVERSE JUST WINKED AT YOU! Reality bent its own rules to deliver this moment. You exist at the intersection of magic ⚡",
    },
}

MULTILINGUAL_PHRASES = {
    "खुश":"joy","खुशी":"joy","प्यार":"joy","आनंद":"joy",
    "गुस्सा":"anger","क्रोध":"anger","नफ़रत":"anger",
    "दुख":"sadness","उदास":"sadness","रोना":"sadness",
    "डर":"fear","भय":"fear","घबराहट":"fear",
    "घृणा":"disgust","नापसंद":"disgust",
    "अचरज":"surprise","हैरान":"surprise",
    "আনন্দ":"joy","খুশি":"joy","ভালোবাসা":"joy",
    "রাগ":"anger","ক্রোধ":"anger",
    "দুঃখ":"sadness","কান্না":"sadness",
    "ভয়":"fear","আতঙ্ক":"fear",
    "ঘৃণা":"disgust",
    "আশ্চর্য":"surprise","বিস্মিত":"surprise",
}

ENGLISH_KEYWORDS = {
    "joy":["happy","happiness","joy","joyful","love","loving","elated","cheerful","excited","wonderful","fantastic","amazing","great","awesome","bliss","blissful","delight","delighted","thrilled","ecstatic","glad","pleased","content","laugh","laughing","smile","smiling","celebrate","celebration","fun","enjoy","enjoying","grateful","gratitude","euphoric","jubilant","radiant","gleeful","overjoyed","beautiful","lovely","divine","perfect","bright","sunshine","warm","peaceful","blessed","lucky","hopeful","light","free"],
    "anger":["angry","anger","furious","rage","mad","irritated","annoyed","frustrated","hate","hatred","outraged","livid","enraged","hostile","bitter","resentful","agitated","infuriated","irate","wrathful","seething","fuming","explosive","violent","temper","scream","yell","aggressive","awful","terrible","horrible","worst","disgusting","fed up","sick of","enough","cannot","unfair"],
    "sadness":["sad","sadness","unhappy","depressed","depression","grief","sorrow","cry","crying","tears","lonely","loneliness","heartbroken","miserable","gloomy","melancholy","hopeless","devastated","despair","mourning","wretched","forlorn","somber","sorrowful","distressed","broken","lost","empty","numb","miss","missing","gone","alone","dark","pain","hurt","ache","weep","abandoned","forgotten","invisible"],
    "fear":["afraid","fear","scared","frightened","terrified","anxious","anxiety","panic","worried","nervous","dread","phobia","horror","terror","uneasy","apprehensive","paranoid","timid","trembling","horrified","petrified","shaken","alarmed","creepy","nightmare","shadow","danger","threat","unsafe","vulnerable","trapped","helpless","overwhelming","cannot breathe","shaking"],
    "disgust":["disgusted","disgust","gross","revolting","repulsed","nauseated","sick","yuck","eww","nasty","vile","repellent","abhorrent","loathe","loathing","offensive","foul","putrid","repugnant","despise","horrible","awful","dreadful","hideous","stench","rotten","corrupt","toxic","unbearable","cannot stand","unacceptable"],
    "surprise":["surprised","surprise","shocked","astonished","amazed","stunned","startled","unexpected","unbelievable","wow","whoa","incredible","mindblowing","astounded","dumbfounded","speechless","bewildered","flabbergasted","dazzled","sudden","impossible","never","cant believe","holy","omg","wait","what","really","seriously","unreal","no way","just found","just heard","just saw"],
}

# ── NLP ────────────────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    if NLP_OK:
        tokens = [t for t in tokens if t not in _stop_words]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def detect_emotion(raw_text):
    scores = {e: 0 for e in EMOTIONS}
    for phrase, emotion in MULTILINGUAL_PHRASES.items():
        if phrase in raw_text:
            scores[emotion] += 2
    tokens = preprocess(raw_text)
    for emotion, keywords in ENGLISH_KEYWORDS.items():
        for token in tokens:
            if token in keywords:
                scores[emotion] += 1
    total = sum(scores.values())
    if total == 0:
        if NLP_OK:
            compound = _sia.polarity_scores(raw_text)["compound"]
        else:
            pos = sum(1 for t in tokens if t in ENGLISH_KEYWORDS["joy"]+ENGLISH_KEYWORDS["surprise"])
            neg = sum(1 for t in tokens if t in ENGLISH_KEYWORDS["sadness"]+ENGLISH_KEYWORDS["anger"]+ENGLISH_KEYWORDS["fear"])
            compound = (pos - neg) / max(len(tokens), 1)
        if   compound >= 0.05: scores["joy"]     = 1
        elif compound <= -0.05: scores["sadness"] = 1
        else: scores["surprise"] = 0.5; scores["joy"] = 0.5
        total = sum(scores.values())
    if total == 0: scores["surprise"] = 1; total = 1
    return {e: round(v/total*100, 1) for e,v in scores.items()}

def dominant_emotion(probs):
    return max(probs, key=probs.get)

# ── CSS ENGINE ────────────────────────────────────────────────────────────────
def inject_css(ed=None):
    c  = ed["color"]    if ed else "#b04aff"
    c2 = ed["color2"]   if ed else "#6af0ff"
    gl = ed["glow"]     if ed else "#b04aff"
    bf = ed["bg_from"]  if ed else "#0d0020"
    bt = ed["bg_to"]    if ed else "#06000f"

    r1, r2, r3 = rgb(c), rgb(c2), rgb(gl)

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Space+Mono:ital@0;1&display=swap');

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stDeployButton"],.stDeployButton,[data-testid="stStatusWidget"],
.viewerBadge_container__1QSob {{ display:none!important; }}
[data-testid="stAppViewContainer"]>.main {{ padding:0!important; }}
section.main>.block-container {{ padding:0!important; max-width:100%!important; }}

/* ── BODY ── */
html,body,[data-testid="stAppViewContainer"] {{
    background:#000008!important;
    min-height:100vh;
    overflow-x:hidden;
}}

/* ── ANIMATED NEBULA ── */
[data-testid="stAppViewContainer"]::before {{
    content:'';
    position:fixed;
    inset:0;
    background:
        radial-gradient(ellipse 90% 70% at 15% 5%,  #1f0040 0%, transparent 55%),
        radial-gradient(ellipse 70% 90% at 85% 95%, #001f40 0%, transparent 55%),
        radial-gradient(ellipse 60% 60% at 50% 50%, rgba({r1},0.08) 0%, transparent 65%),
        radial-gradient(ellipse 40% 40% at 70% 20%, #200010 0%, transparent 50%),
        radial-gradient(ellipse 50% 50% at 30% 80%, #001510 0%, transparent 50%),
        #000008;
    animation: nebulaFlow 18s ease-in-out infinite alternate;
    z-index:0;
    pointer-events:none;
    transition: background 2s ease;
}}
@keyframes nebulaFlow {{
    0%   {{ filter:hue-rotate(0deg)   brightness(1)    saturate(1.3); }}
    33%  {{ filter:hue-rotate(30deg)  brightness(1.15) saturate(1.6); }}
    66%  {{ filter:hue-rotate(-20deg) brightness(0.92) saturate(1.0); }}
    100% {{ filter:hue-rotate(50deg)  brightness(1.1)  saturate(1.5); }}
}}

/* ── NOISE GRAIN ── */
[data-testid="stAppViewContainer"]::after {{
    content:'';
    position:fixed;
    inset:0;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E");
    opacity:.4;
    pointer-events:none;
    z-index:1;
    animation:grain 0.4s steps(2) infinite;
}}
@keyframes grain {{
    0%  {{transform:translate(0,0);}}
    25% {{transform:translate(-2px,1px);}}
    50% {{transform:translate(1px,-2px);}}
    75% {{transform:translate(-1px,2px);}}
    100%{{transform:translate(0,0);}}
}}

/* ── STAR FIELD (JS-generated) ── */
.star-canvas {{ position:fixed;inset:0;z-index:2;pointer-events:none;overflow:hidden; }}
.star {{
    position:absolute;
    border-radius:50%;
    background:white;
    animation:twinkle var(--dur,3s) ease-in-out infinite var(--del,0s);
}}
@keyframes twinkle {{
    0%,100% {{opacity:.15;transform:scale(1);}}
    50%     {{opacity:1; transform:scale(1.6);}}
}}

/* ── SHOOTING STARS ── */
.shoot {{ position:fixed;height:2px;z-index:3;pointer-events:none;border-radius:2px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.95),transparent);
    box-shadow:0 0 8px rgba(255,255,255,0.7); }}
.s1{{width:180px;top:10%;left:-220px;animation:shoot 8s linear infinite 0s;}}
.s2{{width:100px;top:32%;left:-150px;animation:shoot 8s linear infinite 2.8s;opacity:.7;}}
.s3{{width:70px; top:68%;left:-120px;animation:shoot 8s linear infinite 5.5s;opacity:.5;}}
.s4{{width:130px;top:22%;left:-180px;animation:shoot 8s linear infinite 4s;opacity:.6;}}
@keyframes shoot {{
    0%   {{transform:translate(0,0) rotate(18deg);opacity:0;}}
    4%   {{opacity:1;}}
    60%  {{opacity:0;transform:translate(130vw,26vh) rotate(18deg);}}
    100% {{transform:translate(130vw,26vh) rotate(18deg);opacity:0;}}
}}

/* ── FLOATING DUST ── */
.dust {{
    position:fixed;border-radius:50%;pointer-events:none;z-index:3;opacity:0;
    animation:dustRise var(--dur,20s) ease-in-out infinite var(--del,0s);
}}
@keyframes dustRise {{
    0%   {{transform:translateY(100vh) translateX(0) scale(0);opacity:0;}}
    10%  {{opacity:.7;}}
    90%  {{opacity:.2;}}
    100% {{transform:translateY(-10vh) translateX(var(--dx,40px)) scale(2);opacity:0;}}
}}

/* ── MAIN CONTAINER ── */
.cosmos-wrap {{
    position:relative;
    z-index:10;
    max-width:800px;
    margin:28px auto 64px;
    padding:56px 52px 48px;
    background:rgba(4,0,16,0.68);
    backdrop-filter:blur(36px) saturate(1.8) brightness(1.05);
    -webkit-backdrop-filter:blur(36px) saturate(1.8) brightness(1.05);
    border:1px solid rgba({r1},0.28);
    border-radius:36px;
    box-shadow:
        0 0 90px  rgba({r3},0.38),
        0 0 180px rgba({r3},0.18),
        0 0 360px rgba({r3},0.08),
        inset 0 1px 0 rgba(255,255,255,0.14),
        inset 0 -1px 0 rgba({r1},0.08);
    animation:wrapPulse 7s ease-in-out infinite alternate;
    transition:all 1.5s cubic-bezier(0.34,1.56,0.64,1);
}}
@keyframes wrapPulse {{
    0%  {{box-shadow:0 0 90px rgba({r3},0.38), 0 0 180px rgba({r3},0.18), inset 0 1px 0 rgba(255,255,255,0.14);}}
    100%{{box-shadow:0 0 130px rgba({r3},0.58),0 0 260px rgba({r3},0.28), inset 0 1px 0 rgba(255,255,255,0.22);}}
}}
/* Top badge */
.cosmos-wrap::before {{
    content:'✦  ᚷ  ᛃ  ᛇ  ✦';
    position:absolute;
    top:-15px;left:50%;
    transform:translateX(-50%);
    background:rgba(4,0,16,0.95);
    padding:3px 20px;
    border:1px solid rgba({r1},0.45);
    border-radius:30px;
    color:rgba({r1},0.85);
    font-family:'Cinzel Decorative',serif;
    font-size:0.55rem;
    letter-spacing:0.4em;
    white-space:nowrap;
    animation:badgePulse 4s ease-in-out infinite;
}}
@keyframes badgePulse {{
    0%,100%{{box-shadow:0 0 10px rgba({r1},0.3);color:rgba({r1},0.7);}}
    50%    {{box-shadow:0 0 25px rgba({r1},0.7);color:rgba({r1},1);}}
}}

/* ── TITLE ── */
.oracle-title {{
    font-family:'Cinzel Decorative',serif;
    font-size:clamp(2rem,5vw,3.4rem);
    font-weight:900;
    text-align:center;
    background:linear-gradient(135deg,#e8c0ff 0%,{c} 20%,#ffffff 40%,{c2} 60%,#ffb3e6 80%,#e8c0ff 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    background-size:400% 400%;
    animation:titleFlow 5s ease infinite, titleIn 1.1s cubic-bezier(0.34,1.56,0.64,1) both;
    letter-spacing:0.05em;
    line-height:1.15;
    margin-bottom:8px;
}}
@keyframes titleFlow {{
    0%  {{background-position:0% 50%;}}
    50% {{background-position:100% 50%;}}
    100%{{background-position:0% 50%;}}
}}
@keyframes titleIn {{
    from{{opacity:0;transform:scale(0.75) translateY(-24px) rotateX(20deg);}}
    to  {{opacity:1;transform:scale(1)    translateY(0)    rotateX(0deg);}}
}}

.oracle-sub {{
    font-family:'Cormorant Garamond',serif;
    font-style:italic;
    font-size:1.1rem;
    color:rgba(200,170,255,0.72);
    text-align:center;
    letter-spacing:0.16em;
    animation:subIn 1s ease 0.5s both;
}}
@keyframes subIn {{
    from{{opacity:0;transform:translateY(14px);}}
    to  {{opacity:1;transform:translateY(0);}}
}}

.rune-row {{
    text-align:center;
    font-size:1.25rem;
    color:rgba({r1},0.45);
    letter-spacing:0.6em;
    margin:12px 0 28px;
    animation:runeGlow 5s ease-in-out infinite;
}}
@keyframes runeGlow {{
    0%,100%{{opacity:.35;text-shadow:none;}}
    50%    {{opacity:.85;text-shadow:0 0 25px rgba({r1},0.9),0 0 50px rgba({r1},0.4);}}
}}

/* ── DIVIDER ── */
.div-line {{
    position:relative;
    height:1px;
    background:linear-gradient(90deg,transparent,rgba({r1},0.7),rgba({r2},0.7),transparent);
    margin:24px 0;
    animation:divGlow 3s ease-in-out infinite;
}}
.div-line::before {{
    content:'✦';
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    background:rgba(4,0,16,0.95);
    padding:0 12px;
    color:rgba({r1},0.9);
    font-size:0.65rem;
    animation:diamSpin 7s linear infinite;
}}
@keyframes divGlow {{
    0%,100%{{opacity:.5;}}
    50%    {{opacity:1;}}
}}
@keyframes diamSpin {{
    from{{transform:translate(-50%,-50%) rotate(0deg);}}
    to  {{transform:translate(-50%,-50%) rotate(360deg);}}
}}

/* ── TEXTAREA ── */
[data-testid="stTextArea"] label{{display:none!important;}}
[data-testid="stTextArea"] textarea {{
    background:rgba(8,0,24,0.78)!important;
    border:1px solid rgba({r1},0.32)!important;
    border-radius:22px!important;
    color:#eedeff!important;
    font-family:'Cormorant Garamond',serif!important;
    font-size:1.12rem!important;
    line-height:1.75!important;
    padding:22px 26px!important;
    transition:all 0.5s cubic-bezier(0.34,1.56,0.64,1)!important;
    resize:vertical!important;
    letter-spacing:0.025em!important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border-color:rgba({r1},0.95)!important;
    box-shadow:0 0 35px rgba({r3},0.45),0 0 70px rgba({r3},0.18)!important;
    outline:none!important;
}}
[data-testid="stTextArea"] textarea::placeholder {{
    color:rgba(180,140,255,0.35)!important;
    font-style:italic!important;
}}

/* ── BUTTON ── */
[data-testid="stButton"]>button {{
    background:linear-gradient(135deg,{bf},{c}55,{c2}55,{bf})!important;
    background-size:400% 400%!important;
    animation:btnShimmer 3.5s ease infinite!important;
    border:1px solid rgba({r1},0.65)!important;
    border-radius:60px!important;
    color:#fff!important;
    font-family:'Cinzel Decorative',serif!important;
    font-size:1rem!important;
    font-weight:700!important;
    letter-spacing:0.12em!important;
    padding:18px 44px!important;
    width:100%!important;
    cursor:pointer!important;
    transition:all 0.3s cubic-bezier(0.34,1.56,0.64,1)!important;
    box-shadow:0 4px 35px rgba({r3},0.45),inset 0 1px 0 rgba(255,255,255,0.18)!important;
    text-shadow:0 0 25px rgba(255,255,255,0.6)!important;
    position:relative!important;
    overflow:hidden!important;
}}
[data-testid="stButton"]>button:hover {{
    transform:scale(1.06) translateY(-3px)!important;
    box-shadow:0 10px 60px rgba({r3},0.75),0 0 120px rgba({r3},0.35)!important;
    border-color:rgba({r1},1)!important;
}}
[data-testid="stButton"]>button:active {{
    transform:scale(0.96) translateY(1px)!important;
}}
@keyframes btnShimmer {{
    0%  {{background-position:0% 50%;}}
    50% {{background-position:100% 50%;}}
    100%{{background-position:0% 50%;}}
}}

/* ── WARNING ── */
.warn-banner {{
    background:linear-gradient(135deg,rgba(255,100,0,0.08),rgba(255,150,50,0.04));
    border:1px solid rgba(255,150,50,0.42);
    border-radius:20px;
    padding:20px 28px;
    color:#ffcc88;
    font-family:'Cormorant Garamond',serif;
    font-size:1.12rem;
    font-style:italic;
    text-align:center;
    letter-spacing:0.06em;
    line-height:1.7;
    animation:warnPulse 2.5s ease-in-out infinite;
}}
@keyframes warnPulse {{
    0%,100%{{box-shadow:0 0 14px rgba(255,150,50,0.2);}}
    50%    {{box-shadow:0 0 35px rgba(255,150,50,0.55);}}
}}

/* ── PORTAL ── */
.portal-wrap {{
    position:relative;
    width:220px;
    height:220px;
    margin:16px auto;
}}
.pr {{
    position:absolute;
    border-radius:50%;
    border:2px solid transparent;
}}
.pr1 {{
    inset:0;
    border-color:rgba({r1},0.9) transparent rgba({r1},0.4) transparent;
    animation:spinCW 3s linear infinite;
    box-shadow:0 0 24px rgba({r3},0.45),inset 0 0 24px rgba({r3},0.1);
}}
.pr2 {{
    inset:18px;
    border-color:transparent rgba({r2},0.75) transparent rgba({r2},0.35);
    animation:spinCCW 2s linear infinite;
    box-shadow:0 0 18px rgba({r2},0.35);
}}
.pr3 {{
    inset:36px;
    border-style:dashed;
    border-color:rgba(255,255,255,0.22) transparent;
    animation:spinCW 8s linear infinite;
}}
.pr4 {{
    inset:54px;
    border-style:dotted;
    border-color:rgba({r1},0.45) transparent rgba({r1},0.2) transparent;
    animation:spinCCW 5.5s linear infinite;
}}
.pr5 {{
    inset:68px;
    border-color:rgba({r2},0.3) transparent;
    border-style:dashed;
    animation:spinCW 4s linear infinite;
}}
.portal-glow {{
    position:absolute;
    inset:60px;
    border-radius:50%;
    background:radial-gradient(ellipse at center,rgba({r3},0.4),transparent 70%);
    animation:glowBreath 2.2s ease-in-out infinite;
}}
.portal-center {{
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    font-size:4rem;
    z-index:5;
    animation:centerBounce 2.5s ease-in-out infinite;
    filter:drop-shadow(0 0 24px rgba({r3},1));
}}
@keyframes spinCW  {{from{{transform:rotate(0deg);}}to{{transform:rotate(360deg);}}}}
@keyframes spinCCW {{from{{transform:rotate(0deg);}}to{{transform:rotate(-360deg);}}}}
@keyframes glowBreath {{
    0%,100%{{opacity:.5;transform:scale(1);}}
    50%    {{opacity:1; transform:scale(1.4);}}
}}
@keyframes centerBounce {{
    0%,100%{{transform:translate(-50%,-50%) scale(1)   rotate(-5deg);filter:drop-shadow(0 0 24px rgba({r3},1));}}
    25%    {{transform:translate(-50%,-50%) scale(1.12) rotate(3deg); filter:drop-shadow(0 0 40px rgba({r3},1));}}
    75%    {{transform:translate(-50%,-50%) scale(1.06) rotate(-3deg);filter:drop-shadow(0 0 32px rgba({r3},1));}}
}}

/* ── AURA LABEL ── */
.aura-tag {{
    text-align:center;
    font-family:'Space Mono',monospace;
    font-size:0.6rem;
    letter-spacing:0.32em;
    color:rgba({r1},0.72);
    text-transform:uppercase;
    margin:10px 0 4px;
    animation:auraPulse 2.5s ease-in-out infinite;
}}
@keyframes auraPulse {{
    0%,100%{{opacity:.55;text-shadow:none;}}
    50%    {{opacity:1;  text-shadow:0 0 22px rgba({r1},0.85);}}
}}

/* ── EMOTION CARD ── */
.emotion-card {{
    text-align:center;
    padding:20px 12px 16px;
    animation:cardIn 1s cubic-bezier(0.34,1.56,0.64,1) both;
}}
@keyframes cardIn {{
    from{{opacity:0;transform:scale(0.65) rotateY(30deg);}}
    to  {{opacity:1;transform:scale(1)    rotateY(0);}}
}}
.emotion-fairy {{
    font-size:4.5rem;
    display:block;
    animation:fairyDance 3.5s ease-in-out infinite;
    filter:drop-shadow(0 0 28px {c});
}}
@keyframes fairyDance {{
    0%   {{transform:translateY(0)    rotate(-10deg) scale(1);}}
    20%  {{transform:translateY(-18px) rotate(5deg)  scale(1.08);}}
    40%  {{transform:translateY(-10px) rotate(-6deg) scale(1.12);}}
    60%  {{transform:translateY(-22px) rotate(8deg)  scale(1.05);}}
    80%  {{transform:translateY(-6px)  rotate(-4deg) scale(1.09);}}
    100% {{transform:translateY(0)    rotate(-10deg) scale(1);}}
}}

.crystal-row {{
    display:flex;justify-content:center;gap:10px;
    margin:10px 0;
    animation:crystalIn 0.8s ease 0.5s both;
}}
@keyframes crystalIn {{
    from{{opacity:0;transform:scale(0) rotate(180deg);}}
    to  {{opacity:1;transform:scale(1) rotate(0);}}
}}
.xtal {{
    font-size:1.6rem;
    animation:xtalBounce 1.6s ease-in-out infinite;
    filter:drop-shadow(0 0 10px {c});
}}
.xtal:nth-child(1){{animation-delay:0s;}}
.xtal:nth-child(2){{animation-delay:0.18s;}}
.xtal:nth-child(3){{animation-delay:0.36s;font-size:2rem;}}
.xtal:nth-child(4){{animation-delay:0.18s;}}
.xtal:nth-child(5){{animation-delay:0s;}}
@keyframes xtalBounce {{
    0%,100%{{transform:translateY(0) scale(1);}}
    50%    {{transform:translateY(-10px) scale(1.25);}}
}}

.emotion-name {{
    font-family:'Cinzel Decorative',serif;
    font-size:clamp(1.8rem,4.5vw,2.8rem);
    font-weight:900;
    color:{c};
    letter-spacing:0.22em;
    text-transform:uppercase;
    animation:namePulse 2s ease-in-out infinite;
    margin:10px 0 4px;
}}
@keyframes namePulse {{
    0%,100%{{text-shadow:0 0 20px {c},0 0 40px {c}66;}}
    50%    {{text-shadow:0 0 45px {c},0 0 90px {c},0 0 140px {c}33;}}
}}

.emotion-rune {{
    font-size:2.2rem;
    color:rgba({r1},0.45);
    display:block;
    margin:6px 0 10px;
    animation:runeSpin 9s linear infinite;
}}
@keyframes runeSpin {{
    0%  {{transform:rotate(0deg);opacity:.3;}}
    50% {{opacity:.85;}}
    100%{{transform:rotate(360deg);opacity:.3;}}
}}

.emotion-msg {{
    font-family:'Cormorant Garamond',serif;
    font-style:italic;
    font-size:1.12rem;
    color:rgba(225,205,255,0.88);
    max-width:540px;
    margin:0 auto;
    line-height:1.85;
    letter-spacing:0.025em;
    animation:msgIn 0.9s ease 0.6s both;
}}
@keyframes msgIn {{
    from{{opacity:0;transform:translateY(14px);}}
    to  {{opacity:1;transform:translateY(0);}}
}}

/* ── FLOATING PARTICLES ── */
.pfloat-wrap {{
    position:relative;
    height:90px;
    overflow:visible;
    margin:8px 0;
}}
.pfloat {{
    position:absolute;
    font-size:1.9rem;
    animation:pf var(--pdur,3s) ease-in-out infinite var(--pdel,0s);
    filter:drop-shadow(0 0 14px {c}88);
}}
@keyframes pf {{
    0%,100%{{transform:translateY(0) translateX(0) rotate(-6deg) scale(1);   opacity:.7;}}
    33%    {{transform:translateY(-28px) translateX(10px) rotate(9deg) scale(1.25); opacity:1;}}
    66%    {{transform:translateY(-16px) translateX(-8px) rotate(-4deg) scale(0.88);opacity:.8;}}
}}

/* ── SPECTRUM ── */
.spectrum-wrap {{
    background:rgba(255,255,255,0.025);
    border:1px solid rgba({r1},0.18);
    border-radius:24px;
    padding:28px 32px;
    margin-top:8px;
    animation:specIn 0.8s ease 0.4s both;
}}
@keyframes specIn {{
    from{{opacity:0;transform:translateY(22px);}}
    to  {{opacity:1;transform:translateY(0);}}
}}
.spec-title {{
    font-family:'Cinzel Decorative',serif;
    font-size:0.6rem;
    letter-spacing:0.28em;
    color:rgba({r1},0.62);
    text-align:center;
    text-transform:uppercase;
    margin-bottom:22px;
    animation:specTitlePulse 3s ease-in-out infinite;
}}
@keyframes specTitlePulse {{
    0%,100%{{opacity:.55;}}
    50%    {{opacity:1;text-shadow:0 0 15px rgba({r1},0.7);}}
}}

.bar-row {{
    display:flex;
    align-items:center;
    gap:14px;
    margin-bottom:13px;
    animation:barIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}}
.bar-row:nth-child(1){{animation-delay:0.08s;}}
.bar-row:nth-child(2){{animation-delay:0.16s;}}
.bar-row:nth-child(3){{animation-delay:0.24s;}}
.bar-row:nth-child(4){{animation-delay:0.32s;}}
.bar-row:nth-child(5){{animation-delay:0.40s;}}
.bar-row:nth-child(6){{animation-delay:0.48s;}}
@keyframes barIn {{
    from{{opacity:0;transform:translateX(-35px) scale(0.93);}}
    to  {{opacity:1;transform:translateX(0) scale(1);}}
}}
.bar-lbl {{
    font-family:'Cinzel Decorative',serif;
    font-size:0.55rem;
    letter-spacing:0.09em;
    min-width:82px;
    text-align:right;
    line-height:1.4;
}}
.bar-track {{
    flex:1;
    height:13px;
    background:rgba(255,255,255,0.05);
    border-radius:14px;
    overflow:hidden;
    position:relative;
    border:1px solid rgba(255,255,255,0.05);
    box-shadow:inset 0 2px 4px rgba(0,0,0,0.4);
}}
.bar-fill {{
    height:100%;
    border-radius:14px;
    position:relative;
    overflow:hidden;
    animation:fillGrow 1.4s cubic-bezier(0.34,1.56,0.64,1) both;
}}
@keyframes fillGrow {{
    from{{width:0!important;}}
}}
.bar-fill::after {{
    content:'';
    position:absolute;
    top:0;left:-80%;
    width:60%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.55),transparent);
    animation:barShine 2.2s ease-in-out infinite;
}}
@keyframes barShine {{
    0%  {{left:-80%;}}
    100%{{left:180%;}}
}}
.bar-pct {{
    font-family:'Space Mono',monospace;
    font-size:0.68rem;
    min-width:36px;
    color:rgba(200,170,255,0.6);
}}

/* ── FOOTER ── */
.cosmic-footer {{
    margin-top:36px;
    padding:20px 32px;
    background:linear-gradient(135deg,rgba({r1},0.07),rgba({r2},0.04));
    border:1px solid rgba({r1},0.3);
    border-radius:20px;
    text-align:center;
    font-family:'Cinzel Decorative',serif;
    font-size:0.68rem;
    letter-spacing:0.28em;
    color:rgba(200,170,255,0.62);
    animation:footerGlow 5s ease-in-out infinite alternate;
    position:relative;
    overflow:hidden;
}}
.cosmic-footer::before {{
    content:'';
    position:absolute;
    top:-50%;left:-50%;width:200%;height:200%;
    background:conic-gradient(from 0deg at 50% 50%,
        transparent 0%,rgba({r1},0.06) 25%,
        transparent 50%,rgba({r2},0.06) 75%,transparent 100%);
    animation:footerSpin 10s linear infinite;
}}
@keyframes footerSpin {{from{{transform:rotate(0deg);}}to{{transform:rotate(360deg);}}}}
@keyframes footerGlow {{
    0%  {{box-shadow:0 0 18px rgba({r3},0.22);}}
    100%{{box-shadow:0 0 50px rgba({r3},0.55),0 0 100px rgba({r3},0.2);}}
}}
.footer-inner {{position:relative;z-index:2;}}
.footer-sub {{
    opacity:.4;
    font-size:.5rem;
    letter-spacing:.18em;
    display:block;
    margin-top:6px;
    font-family:'Space Mono',monospace;
}}

/* ── SPINNER ── */
[data-testid="stSpinner"] p {{
    font-family:'Cormorant Garamond',serif!important;
    font-style:italic!important;
    color:rgba(200,170,255,0.85)!important;
    font-size:1.12rem!important;
    letter-spacing:0.1em!important;
    animation:spinPulse 1.5s ease-in-out infinite!important;
}}
@keyframes spinPulse {{
    0%,100%{{opacity:.55;}}
    50%    {{opacity:1;}}
}}
</style>

<!-- BACKGROUND LAYERS -->
<div class="star-canvas" id="starCanvas"></div>
<div class="shoot s1"></div><div class="shoot s2"></div>
<div class="shoot s3"></div><div class="shoot s4"></div>

<script>
(function(){{
    // Stars
    const c = document.getElementById('starCanvas');
    if(!c) return;
    for(let i=0;i<150;i++){{
        const s=document.createElement('div');
        s.className='star';
        const sz=Math.random()*2.8+0.4;
        const hue=Math.random()*80+200;
        s.style.cssText=`width:${{sz}}px;height:${{sz}}px;top:${{Math.random()*100}}%;left:${{Math.random()*100}}%;--dur:${{(Math.random()*5+1.5).toFixed(1)}}s;--del:-${{(Math.random()*6).toFixed(1)}}s;background:hsl(${{hue}},80%,${{Math.random()*25+68}}%);`;
        c.appendChild(s);
    }}
    // Dust
    const body=document.querySelector('[data-testid="stAppViewContainer"]')||document.body;
    const glowR='{r3}';
    for(let i=0;i<25;i++){{
        const d=document.createElement('div');
        d.className='dust';
        const sz=Math.random()*5+2;
        d.style.cssText=`width:${{sz}}px;height:${{sz}}px;left:${{Math.random()*100}}%;bottom:0;background:rgba(${{glowR}},${{(Math.random()*0.45+0.1).toFixed(2)}});--dur:${{(Math.random()*16+10).toFixed(1)}}s;--del:-${{(Math.random()*22).toFixed(1)}}s;--dx:${{((Math.random()-0.5)*100).toFixed(0)}}px;box-shadow:0 0 ${{sz*3}}px rgba(${{glowR}},0.6);`;
        body.appendChild(d);
    }}
}})();
</script>
""", unsafe_allow_html=True)

# ── Render Helpers ────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
<div class="oracle-title">🌌 Vibe Oracle</div>
<div class="oracle-sub">The cosmos decodes the language of your soul</div>
<div class="rune-row">ᚷ &nbsp; ✦ &nbsp; ᛃ &nbsp; ✦ &nbsp; ᛇ &nbsp; ✦ &nbsp; ᚾ &nbsp; ✦ &nbsp; ᚦ</div>
""", unsafe_allow_html=True)

def render_divider():
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)

def render_portal(ed):
    st.markdown(f"""
<div class="portal-wrap">
    <div class="pr pr1"></div>
    <div class="pr pr2"></div>
    <div class="pr pr3"></div>
    <div class="pr pr4"></div>
    <div class="pr pr5"></div>
    <div class="portal-glow"></div>
    <div class="portal-center">{ed["moon"]}</div>
</div>
<div class="aura-tag">⟡ &nbsp; {ed["aura"]} &nbsp; ⟡</div>
""", unsafe_allow_html=True)

def render_particles(ed):
    p = ed["particles"]
    positions = [5, 14, 24, 35, 48, 60, 72, 82, 90, 96]
    tops      = [30, 10, 55, 20, 45, 15, 60, 25, 40, 5]
    durs      = [2.2, 3.0, 2.6, 3.4, 2.0, 3.1, 2.8, 2.4, 3.2, 2.7]
    dels      = [0, 0.5, 1.0, 0.3, 1.4, 0.7, 0.2, 1.1, 0.8, 1.6]
    html = '<div class="pfloat-wrap">'
    for i, emoji in enumerate(p[:10]):
        html += f'<span class="pfloat" style="left:{positions[i]}%;top:{tops[i]}%;--pdur:{durs[i]}s;--pdel:-{dels[i]}s">{emoji}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_emotion_card(emotion, ed):
    cr = ed["crystal"]
    st.markdown(f"""
<div class="emotion-card">
    <span class="emotion-fairy">{ed["fairy"]}</span>
    <div class="crystal-row">
        <span class="xtal">{cr}</span>
        <span class="xtal">{cr}</span>
        <span class="xtal">{ed["emoji"]}</span>
        <span class="xtal">{cr}</span>
        <span class="xtal">{cr}</span>
    </div>
    <div class="emotion-name">{emotion}</div>
    <span class="emotion-rune">{ed["rune"]}</span>
    <div class="emotion-msg">{ed["message"]}</div>
</div>
""", unsafe_allow_html=True)

def render_spectrum(probs):
    sorted_e = sorted(probs.items(), key=lambda x: -x[1])
    html = '<div class="spectrum-wrap"><div class="spec-title">✦ &nbsp;&nbsp; Emotional Spectrum &nbsp;&nbsp; ✦</div>'
    for emotion, pct in sorted_e:
        ed = EMOTIONS[emotion]
        c, c2 = ed["color"], ed["color2"]
        w = max(pct, 0.8)
        html += f"""
<div class="bar-row">
  <div class="bar-lbl" style="color:{c}">{ed['emoji']}&nbsp;{emotion.capitalize()}</div>
  <div class="bar-track">
    <div class="bar-fill" style="width:{w}%;background:linear-gradient(90deg,{c}44,{c},{c2})"></div>
  </div>
  <div class="bar-pct">{pct:.0f}%</div>
</div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
<div class="cosmic-footer">
  <div class="footer-inner">
    ✦ &nbsp; Vibe decoded by the Cosmic Fairy &nbsp; ✦
    <span class="footer-sub">POWERED BY STARLIGHT &amp; ANCIENT NLP RUNES &amp; COSMIC VIBRATIONS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ed = st.session_state.get("last_ed", None)
    inject_css(ed=ed)

    st.markdown('<div class="cosmos-wrap">', unsafe_allow_html=True)
    render_header()
    render_divider()

    user_text = st.text_area(
        "soul",
        placeholder="Pour your soul here… the stars are listening ✨",
        height=155,
        key="soul_input",
        label_visibility="collapsed",
    )

    submitted = st.button("🔮 Reveal the Vibe")

    if submitted:
        raw = user_text.strip()
        if not raw:
            st.markdown("""
<div class="warn-banner">
🌙 The cosmos received only silence…<br>
<span style="font-size:.9rem;opacity:.8">Speak your truth — even a whisper reshapes the stars</span>
</div>""", unsafe_allow_html=True)
        else:
            with st.spinner("✨ Consulting the cosmos… aligning your stars…"):
                probs = detect_emotion(raw)
            dom = dominant_emotion(probs)
            st.session_state["last_ed"]    = EMOTIONS[dom]
            st.session_state["last_probs"] = probs
            st.session_state["last_dom"]   = dom
            st.session_state["has_result"] = True

    if st.session_state.get("has_result"):
        ed2   = st.session_state.get("last_ed")
        probs = st.session_state.get("last_probs", {})
        dom   = st.session_state.get("last_dom", "joy")
        if ed2 and probs:
            render_divider()
            render_portal(ed2)
            render_particles(ed2)
            render_emotion_card(dom, ed2)
            render_divider()
            render_spectrum(probs)

    render_divider()
    render_footer()
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

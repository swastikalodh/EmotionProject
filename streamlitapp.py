"""
🌑 DREAMWEAVER — Cosmic Dream Decoder
Type your dream. The cosmos decodes its meaning.
Reveals: Archetype · Prophecy · Shadow · Symbol · Energy

pip install streamlit nltk
streamlit run dreamweaver.py
"""

import streamlit as st
import re, random, math, json, hashlib

st.set_page_config(page_title="🌑 DREAMWEAVER", page_icon="🌑", layout="wide",
                   initial_sidebar_state="collapsed")

# ── NLP ───────────────────────────────────────────────────────────────────────
try:
    import nltk
    for pkg, path in [("stopwords","corpora/stopwords"),("wordnet","corpora/wordnet"),("punkt","tokenizers/punkt")]:
        try: nltk.data.find(path)
        except: nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _SW = set(stopwords.words("english")); _L = WordNetLemmatizer(); NLP=True
except: NLP=False

# ── DREAM ARCHETYPES ──────────────────────────────────────────────────────────
ARCHETYPES = {
    "The Wanderer": {
        "keys": ["road","path","walk","travel","lost","journey","map","direction","wandering","leaving","moving","going","running","forest","desert","city","unknown"],
        "colors": ["#00FFFF","#0088FF","#0000FF"],
        "glyph": "⟋",
        "shadow": "The fear of never arriving, of motion without purpose.",
        "prophecy": "A threshold is near. The crossroads you've been circling will finally reveal its true sign. Step forward without a map — the path only appears beneath moving feet.",
        "element": "WIND", "planet": "Mercury", "number": 7,
        "ritual": "Write the destination you're afraid to name.",
    },
    "The Shadow Self": {
        "keys": ["dark","darkness","shadow","monster","creature","chased","running","fear","nightmare","black","hide","hiding","evil","demon","ghost","death","dying"],
        "colors": ["#FF0080","#800080","#300030"],
        "glyph": "◐",
        "shadow": "The parts of yourself you refuse to look at directly.",
        "prophecy": "What hunts you in the dark is not your enemy — it is a piece of yourself waiting to be reclaimed. Turn around. Face it. Ask its name.",
        "element": "VOID", "planet": "Pluto", "number": 0,
        "ritual": "Draw or describe the thing that chases you. Give it a name.",
    },
    "The Oracle": {
        "keys": ["light","glow","vision","see","saw","message","voice","speaking","heard","ancient","temple","star","stars","sky","above","higher","truth","knowing"],
        "colors": ["#FFD700","#FF8C00","#FFFF00"],
        "glyph": "◎",
        "shadow": "The terror of knowing what you cannot yet prove.",
        "prophecy": "A signal is being broadcast directly to you from the deep structure of reality. You already know the answer. The dream is confirming, not revealing.",
        "element": "FIRE", "planet": "Sun", "number": 1,
        "ritual": "Write three things you already know but haven't admitted.",
    },
    "The Lover": {
        "keys": ["love","lover","kiss","touch","embrace","together","heart","warm","beautiful","person","someone","hand","holding","close","feel","feeling","connection","intimate"],
        "colors": ["#FF69B4","#FF1493","#FF4500"],
        "glyph": "♡",
        "shadow": "The grief of connection withheld, love left unspoken.",
        "prophecy": "Someone or something is moving toward you. Open your hands — stop gripping what was. The heart is not a museum. It is a river.",
        "element": "WATER", "planet": "Venus", "number": 2,
        "ritual": "Say aloud what you have been afraid to feel.",
    },
    "The Architect": {
        "keys": ["house","home","room","building","door","window","wall","structure","floor","ceiling","stairs","basement","attic","space","place","inside","outside","city","tower"],
        "colors": ["#00FF88","#00FFAA","#00AA66"],
        "glyph": "⬡",
        "shadow": "The prison of a self that has been overbuilt for safety.",
        "prophecy": "The structure of your life is being reviewed from the inside. A room you have locked will be opened. Something that felt like shelter may be a cage.",
        "element": "EARTH", "planet": "Saturn", "number": 4,
        "ritual": "What room inside you have you not entered in years?",
    },
    "The Transformer": {
        "keys": ["fire","burning","change","transform","new","old","end","begin","break","broken","grow","growing","birth","death","rebirth","water","flood","storm","lightning","explosion"],
        "colors": ["#FF4500","#FF0000","#FF6600"],
        "glyph": "⚡",
        "shadow": "Resistance to the self that is trying to be born.",
        "prophecy": "You are in the middle of becoming. The burning is not destruction — it is the cost of the new form. Something is ending so completely it cannot be mourned. Only celebrated.",
        "element": "FIRE", "planet": "Mars", "number": 9,
        "ritual": "Name one thing that must end so something else can live.",
    },
    "The Deep One": {
        "keys": ["ocean","sea","water","deep","swim","swimming","underwater","fish","wave","waves","lake","river","rain","floating","drowning","sinking","submerge","abyss"],
        "colors": ["#0044FF","#0099FF","#00CCFF"],
        "glyph": "◌",
        "shadow": "The terror of depth, of what lives in the unconscious ocean.",
        "prophecy": "You are being invited to dive. The surface world has given you all it can. What you need now is in the depths — in the feelings you have been skimming across.",
        "element": "WATER", "planet": "Neptune", "number": 3,
        "ritual": "Sit in silence for 3 minutes. Notice what rises.",
    },
    "The Cosmic Child": {
        "keys": ["fly","flying","float","floating","space","moon","sun","stars","galaxy","infinite","dream","dreaming","magic","magical","strange","weird","impossible","wonder","sky","cloud"],
        "colors": ["#AA00FF","#FF00FF","#FF00AA"],
        "glyph": "✦",
        "shadow": "The grief of a wonder that the world tried to cure you of.",
        "prophecy": "You have access to frequencies others cannot receive. The strangeness of this dream is a feature, not a bug. Reality is wider than the version you've been handed.",
        "element": "AETHER", "planet": "Uranus", "number": 11,
        "ritual": "Do one impossible thing today. Even in imagination.",
    },
}

SYMBOLS = {
    "snake":  ("Kundalini awakening. Hidden wisdom coiling upward.", "#00FF88"),
    "door":   ("A threshold. An irreversible decision approaching.", "#FFD700"),
    "mirror": ("Self-confrontation. The reflection holds a truth.", "#00FFFF"),
    "falling":("Surrender. The ego releasing its grip on control.", "#FF4500"),
    "teeth":  ("Power and communication. Something unsaid is costing you.", "#FF69B4"),
    "bird":   ("A message in transit. The soul taking flight.", "#AAFFFF"),
    "blood":  ("Vital force. Sacrifice. The price of transformation.", "#FF0040"),
    "eye":    ("Cosmic witness. You are being seen by something vast.", "#FFD700"),
    "key":    ("Access to locked knowledge. A solution already in hand.", "#AA88FF"),
    "baby":   ("New potential. A project or self not yet born.", "#FF99CC"),
    "storm":  ("Emotional reckoning. The clearing after chaos.", "#4488FF"),
    "gold":   ("The authentic self emerging from shadow material.", "#FFD700"),
    "tower":  ("Structures built on false foundations being dismantled.", "#FF4500"),
    "wolf":   ("Instinct restored. The wild self reclaiming its territory.", "#AAAAFF"),
    "clock":  ("Urgency. A window that will not stay open indefinitely.", "#00FFCC"),
    "void":   ("The fertile emptiness before creation.", "#8800FF"),
}

COSMIC_LINES = [
    "The dream does not belong to you — you belong to the dream.",
    "Sleep is where the soul files its reports.",
    "Every nightmare is a guardian in disguise.",
    "The unconscious speaks in metaphor because the truth is too large for words.",
    "You dreamed this so you would know.",
    "The cosmos annotates your days while you sleep.",
    "What you cannot face awake, you face in dreams — in costume.",
    "The dream is the universe's first draft of your next chapter.",
]

def tokens(text):
    t = text.lower(); t = re.sub(r"[^\w\s]"," ",t); words = t.split()
    if NLP: words = [_L.lemmatize(w) for w in words if w not in _SW]
    return words

def decode_dream(text):
    raw = text.lower()
    toks = tokens(text)
    all_words = set(toks) | set(raw.split())

    # Score archetypes
    scores = {}
    for name, data in ARCHETYPES.items():
        sc = sum(1 for k in data["keys"] if k in all_words)
        if sc > 0: scores[name] = sc

    if not scores:
        # Hash-based deterministic fallback
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        name = list(ARCHETYPES.keys())[h % len(ARCHETYPES)]
        scores = {name: 1}

    # Primary archetype
    primary = max(scores, key=scores.get)

    # Secondary (if any)
    secondary = None
    if len(scores) > 1:
        s2 = sorted(scores, key=scores.get, reverse=True)
        secondary = s2[1]

    # Detect symbols
    found_symbols = []
    for sym, (meaning, color) in SYMBOLS.items():
        if sym in raw or sym in toks:
            found_symbols.append((sym, meaning, color))

    # Random cosmic line seeded by text
    seed = sum(ord(c) for c in text)
    cosmic = COSMIC_LINES[seed % len(COSMIC_LINES)]

    # Dream intensity (word count based)
    intensity = min(100, max(20, len(text.split()) * 2))

    return {
        "primary": primary,
        "secondary": secondary,
        "arch": ARCHETYPES[primary],
        "symbols": found_symbols[:4],
        "cosmic": cosmic,
        "intensity": intensity,
        "word_count": len(text.split()),
    }

# ── HTML / CANVAS ENGINE ──────────────────────────────────────────────────────
def build_page(result=None, warn=False):
    res_json = "null"
    if result:
        arch = result["arch"]
        res_json = json.dumps({
            "primary":    result["primary"],
            "secondary":  result["secondary"],
            "colors":     arch["colors"],
            "glyph":      arch["glyph"],
            "shadow":     arch["shadow"],
            "prophecy":   arch["prophecy"],
            "element":    arch["element"],
            "planet":     arch["planet"],
            "number":     arch["number"],
            "ritual":     arch["ritual"],
            "cosmic":     result["cosmic"],
            "intensity":  result["intensity"],
            "symbols":    [[s,m,c] for s,m,c in result["symbols"]],
            "word_count": result["word_count"],
        })

    warn_anim = "animation:toastIn 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;" if warn else ""

    return f"""
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stDeployButton"],.stDeployButton,[data-testid="stStatusWidget"]
{{display:none!important;}}
[data-testid="stAppViewContainer"]>.main{{padding:0!important;}}
section.main>.block-container{{padding:0!important;max-width:100%!important;}}
html,body,[data-testid="stAppViewContainer"]{{
    background:#000!important;overflow:hidden!important;
    height:100vh!important;width:100vw!important;
}}
[data-testid="stVerticalBlock"]{{gap:0!important;}}

/* FORM — fixed at bottom */
[data-testid="stForm"]{{
    position:fixed!important;bottom:0!important;left:50%!important;
    transform:translateX(-50%)!important;z-index:99999!important;
    width:min(740px,96vw)!important;background:transparent!important;
    border:none!important;padding:0 0 28px!important;
}}
[data-testid="stTextArea"] label{{display:none!important;}}
[data-testid="stTextArea"]{{
    position:fixed!important;bottom:96px!important;left:50%!important;
    transform:translateX(-50%)!important;z-index:99999!important;
    width:min(740px,96vw)!important;
}}
[data-testid="stTextArea"] textarea{{
    background:rgba(5,0,15,0.88)!important;
    border:1px solid rgba(255,255,255,0.1)!important;
    border-radius:18px!important;color:#e8d8ff!important;
    font-family:Georgia,serif!important;font-style:italic!important;
    font-size:1rem!important;line-height:1.75!important;
    padding:16px 20px!important;resize:none!important;outline:none!important;
    backdrop-filter:blur(40px)!important;height:90px!important;
    transition:all 0.4s ease!important;
    box-shadow:0 0 40px rgba(100,0,255,0.12)!important;
}}
[data-testid="stTextArea"] textarea:focus{{
    border-color:rgba(180,100,255,0.6)!important;
    box-shadow:0 0 60px rgba(150,0,255,0.35),0 0 120px rgba(100,0,255,0.15)!important;
}}
[data-testid="stTextArea"] textarea::placeholder{{color:rgba(180,140,255,0.3)!important;}}
[data-testid="stFormSubmitButton"]{{
    position:fixed!important;bottom:28px!important;left:50%!important;
    transform:translateX(-50%)!important;z-index:99999!important;
    width:min(740px,96vw)!important;
}}
[data-testid="stFormSubmitButton"] button{{
    width:100%!important;padding:15px 40px!important;border:none!important;
    border-radius:50px!important;font-size:0.85rem!important;font-weight:800!important;
    letter-spacing:0.28em!important;text-transform:uppercase!important;
    cursor:pointer!important;color:#fff!important;
    background:linear-gradient(135deg,#3300aa,#8800ff,#ff00aa,#ff4400,#ffcc00)!important;
    background-size:400% 400%!important;animation:btnPulse 4s ease infinite!important;
    box-shadow:0 0 50px rgba(140,0,255,0.5)!important;
    transition:transform 0.2s,box-shadow 0.2s!important;
}}
[data-testid="stFormSubmitButton"] button:hover{{
    transform:scale(1.04) translateY(-2px)!important;
    box-shadow:0 0 90px rgba(180,0,255,0.8)!important;
}}
[data-testid="stSpinner"]{{display:none!important;}}
@keyframes btnPulse{{
    0%{{background-position:0% 50%;}}50%{{background-position:100% 50%;}}100%{{background-position:0% 50%;}}
}}
</style>

<!-- FULL SCREEN EXPERIENCE -->
<div id="DW" style="position:fixed;inset:0;width:100vw;height:100vh;overflow:hidden;z-index:500;pointer-events:none;">

  <!-- CANVASES -->
  <canvas id="bgC" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
  <canvas id="stC" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>

  <!-- MAIN UI -->
  <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;padding:clamp(20px,4vh,50px) 20px 220px;overflow-y:auto;">

    <!-- HEADER -->
    <div style="text-align:center;margin-bottom:clamp(16px,3vh,32px);">
      <div id="dwTitle" style="font-size:clamp(2rem,5.5vw,4.2rem);font-weight:900;letter-spacing:0.1em;line-height:1;user-select:none;cursor:default;"></div>
      <div style="font-size:clamp(0.5rem,1vw,0.7rem);letter-spacing:0.5em;color:rgba(255,255,255,0.22);margin-top:10px;text-transform:uppercase;font-family:monospace;">Cosmic Dream Decoder · Archetype Revelator</div>
      <div id="moonPhase" style="font-size:2rem;margin-top:10px;animation:moonSpin 12s linear infinite;display:inline-block;filter:drop-shadow(0 0 20px rgba(200,150,255,0.8));">🌑</div>
    </div>

    <!-- RESULT AREA -->
    <div id="resultWrap" style="display:none;width:min(760px,96vw);animation:resultSlide 1s cubic-bezier(0.34,1.56,0.64,1) both;">

      <!-- COSMIC QUOTE -->
      <div id="cosmicQuote" style="
        text-align:center;font-family:Georgia,serif;font-style:italic;
        font-size:clamp(0.8rem,1.5vw,1rem);color:rgba(200,180,255,0.6);
        letter-spacing:0.08em;margin-bottom:20px;line-height:1.8;
        animation:fadeUp 0.8s ease 0.2s both;opacity:0;
      "></div>

      <!-- ARCHETYPE CARD -->
      <div id="archCard" style="
        background:rgba(10,0,30,0.75);
        border-radius:28px;padding:clamp(20px,3vw,36px);
        margin-bottom:16px;position:relative;overflow:hidden;
        backdrop-filter:blur(30px);
        border:1px solid rgba(255,255,255,0.07);
        animation:fadeUp 0.8s ease 0.1s both;opacity:0;
      ">
        <div id="archBg" style="position:absolute;inset:0;border-radius:28px;opacity:0.08;pointer-events:none;"></div>
        <div id="archGlow" style="position:absolute;top:-60px;right:-60px;width:200px;height:200px;border-radius:50%;opacity:0.15;filter:blur(60px);pointer-events:none;"></div>

        <!-- TOP ROW -->
        <div style="display:flex;align-items:flex-start;gap:20px;margin-bottom:20px;position:relative;z-index:1;">
          <div id="archGlyph" style="font-size:clamp(3rem,8vw,5rem);line-height:1;animation:glyphFloat 4s ease-in-out infinite;flex-shrink:0;"></div>
          <div style="flex:1;">
            <div style="font-size:0.52rem;letter-spacing:0.4em;color:rgba(255,255,255,0.25);text-transform:uppercase;font-family:monospace;margin-bottom:6px;">PRIMARY ARCHETYPE</div>
            <div id="archName" style="font-size:clamp(1.4rem,3.5vw,2.4rem);font-weight:900;letter-spacing:0.1em;line-height:1.1;margin-bottom:8px;"></div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
              <div id="tagElement" style="font-size:0.52rem;letter-spacing:0.2em;padding:4px 12px;border-radius:20px;border:1px solid;font-family:monospace;opacity:0.7;"></div>
              <div id="tagPlanet"  style="font-size:0.52rem;letter-spacing:0.2em;padding:4px 12px;border-radius:20px;border:1px solid;font-family:monospace;opacity:0.7;"></div>
              <div id="tagNumber"  style="font-size:0.52rem;letter-spacing:0.2em;padding:4px 12px;border-radius:20px;border:1px solid;font-family:monospace;opacity:0.7;"></div>
            </div>
          </div>
        </div>

        <!-- DIVIDER -->
        <div id="archDiv" style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);margin:0 0 20px;position:relative;z-index:1;"></div>

        <!-- PROPHECY -->
        <div style="position:relative;z-index:1;margin-bottom:20px;">
          <div style="font-size:0.5rem;letter-spacing:0.35em;color:rgba(255,255,255,0.25);text-transform:uppercase;font-family:monospace;margin-bottom:10px;">◈ &nbsp; THE PROPHECY</div>
          <div id="prophecyText" style="font-family:Georgia,serif;font-size:clamp(0.88rem,1.5vw,1.05rem);line-height:1.85;color:rgba(230,215,255,0.88);font-style:italic;"></div>
        </div>

        <!-- SHADOW -->
        <div style="position:relative;z-index:1;margin-bottom:20px;padding:16px 20px;background:rgba(0,0,0,0.3);border-radius:14px;border-left:3px solid;">
          <div id="shadowBorder" style="display:none;"></div>
          <div style="font-size:0.5rem;letter-spacing:0.35em;color:rgba(255,255,255,0.2);text-transform:uppercase;font-family:monospace;margin-bottom:8px;">◐ &nbsp; YOUR SHADOW</div>
          <div id="shadowText" style="font-size:0.88rem;color:rgba(200,180,255,0.65);line-height:1.7;font-style:italic;font-family:Georgia,serif;"></div>
        </div>

        <!-- RITUAL -->
        <div style="position:relative;z-index:1;padding:14px 20px;background:rgba(255,255,255,0.03);border-radius:14px;border:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:0.5rem;letter-spacing:0.35em;color:rgba(255,255,255,0.2);text-transform:uppercase;font-family:monospace;margin-bottom:8px;">✦ &nbsp; YOUR RITUAL</div>
          <div id="ritualText" style="font-size:0.88rem;color:rgba(200,200,255,0.75);line-height:1.6;font-family:Georgia,serif;"></div>
        </div>
      </div>

      <!-- SYMBOLS ROW -->
      <div id="symbolsWrap" style="display:none;margin-bottom:16px;animation:fadeUp 0.8s ease 0.4s both;opacity:0;">
        <div style="font-size:0.5rem;letter-spacing:0.4em;color:rgba(255,255,255,0.2);text-transform:uppercase;font-family:monospace;text-align:center;margin-bottom:12px;">◈ &nbsp; DREAM SYMBOLS DETECTED &nbsp; ◈</div>
        <div id="symbolsGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;"></div>
      </div>

      <!-- INTENSITY BAR -->
      <div style="animation:fadeUp 0.8s ease 0.5s both;opacity:0;background:rgba(10,0,30,0.6);border-radius:18px;padding:18px 24px;backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.06);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
          <div style="font-size:0.5rem;letter-spacing:0.35em;color:rgba(255,255,255,0.22);text-transform:uppercase;font-family:monospace;">◈ &nbsp; DREAM INTENSITY</div>
          <div id="intensityPct" style="font-size:0.65rem;font-family:monospace;color:rgba(255,255,255,0.35);"></div>
        </div>
        <div style="height:8px;background:rgba(255,255,255,0.05);border-radius:8px;overflow:hidden;">
          <div id="intensityBar" style="height:100%;border-radius:8px;animation:barGrow 1.5s cubic-bezier(0.34,1.56,0.64,1) 0.6s both;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:-60%;width:50%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.5),transparent);animation:shine 2s ease-in-out infinite;"></div>
          </div>
        </div>
        <div id="secondaryLine" style="margin-top:12px;font-size:0.6rem;letter-spacing:0.15em;color:rgba(255,255,255,0.2);font-family:monospace;text-align:center;display:none;"></div>
      </div>

    </div>

    <!-- IDLE STATE (no result yet) -->
    <div id="idleState" style="text-align:center;margin-top:clamp(10px,4vh,40px);">
      <div id="idleGlyph" style="font-size:clamp(3rem,8vw,6rem);animation:idleFloat 5s ease-in-out infinite;filter:drop-shadow(0 0 40px rgba(140,0,255,0.6));margin-bottom:20px;">🌑</div>
      <div style="font-size:clamp(0.7rem,1.5vw,0.9rem);letter-spacing:0.3em;color:rgba(200,160,255,0.35);text-transform:uppercase;font-family:monospace;line-height:2;">Describe your dream below<br>and receive your cosmic reading</div>
      <div style="margin-top:20px;display:flex;justify-content:center;gap:16px;font-size:1.5rem;animation:iconDrift 3s ease-in-out infinite;">
        <span style="animation:iconDrift 3s ease-in-out infinite 0s;">🌙</span>
        <span style="animation:iconDrift 3s ease-in-out infinite 0.5s;">⭐</span>
        <span style="animation:iconDrift 3s ease-in-out infinite 1s;">🔮</span>
        <span style="animation:iconDrift 3s ease-in-out infinite 1.5s;">✨</span>
        <span style="animation:iconDrift 3s ease-in-out infinite 2s;">🌌</span>
      </div>
    </div>

  </div>

  <!-- WARNING TOAST -->
  <div id="warnToast" style="
    position:fixed;bottom:180px;left:50%;transform:translateX(-50%) translateY(80px);
    background:rgba(200,50,0,0.15);border:1px solid rgba(255,120,50,0.4);
    border-radius:50px;padding:12px 28px;color:#ffaa66;
    font-size:0.75rem;letter-spacing:0.15em;backdrop-filter:blur(20px);
    transition:transform 0.5s cubic-bezier(0.34,1.56,0.64,1);
    white-space:nowrap;z-index:200000;pointer-events:none;
    {warn_anim}
  ">🌑 &nbsp; The dream journal awaits your words &nbsp; 🌑</div>

</div>

<style>
@keyframes moonSpin{{from{{transform:rotate(0deg);}}to{{transform:rotate(360deg);}}}}
@keyframes resultSlide{{from{{opacity:0;transform:translateY(40px);}}to{{opacity:1;transform:translateY(0);}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(20px);}}to{{opacity:1;transform:translateY(0);}}}}
@keyframes glyphFloat{{0%,100%{{transform:translateY(0) rotate(-5deg);}}50%{{transform:translateY(-14px) rotate(5deg);}}}}
@keyframes idleFloat{{0%,100%{{transform:translateY(0) scale(1);}}50%{{transform:translateY(-20px) scale(1.05);}}}}
@keyframes iconDrift{{0%,100%{{transform:translateY(0);}}50%{{transform:translateY(-8px);}}}}
@keyframes barGrow{{from{{width:0!important;}}}}
@keyframes shine{{0%{{left:-60%;}}100%{{left:160%;}}}}
@keyframes toastIn{{to{{transform:translateX(-50%) translateY(0);}}}}
@keyframes toastOut{{to{{transform:translateX(-50%) translateY(80px);}}}}
@keyframes titleWave{{0%,100%{{transform:translateY(0) scaleX(1);}}50%{{transform:translateY(-5px) scaleX(1.02);}}}}
</style>

<script>
(function(){{
const RESULT = {res_json};

// ── CANVAS ────────────────────────────────────────────────────────────────────
const bgC=document.getElementById('bgC'), stC=document.getElementById('stC');
const bgX=bgC.getContext('2d'), stX=stC.getContext('2d');
let W,H,T=0,mx=window.innerWidth/2,my=window.innerHeight/2;
let pal=['#3300aa','#8800ff','#ff00aa','#4400cc'];
let tPal=pal.slice(), lT=0;

function rsz(){{W=bgC.width=stC.width=window.innerWidth;H=bgC.height=stC.height=window.innerHeight;}}
rsz(); window.addEventListener('resize',rsz);
document.addEventListener('mousemove',e=>{{mx=e.clientX;my=e.clientY;spawnMouse();}});
document.addEventListener('touchmove',e=>{{mx=e.touches[0].clientX;my=e.touches[0].clientY;}},{{passive:true}});
document.addEventListener('click',e=>{{
    if(['BUTTON','TEXTAREA','INPUT'].includes(e.target.tagName)) return;
    burst(e.clientX,e.clientY,25);
}});

// Color utils
function h2r(h){{h=h.replace('#','');if(h.length===3)h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}}
function stepPal(){{lT=Math.min(lT+0.01,1);for(let i=0;i<4;i++){{const A=h2r(pal[i]),B=h2r(tPal[i]);pal[i]='#'+[A[0]+(B[0]-A[0])*lT,A[1]+(B[1]-A[1])*lT,A[2]+(B[2]-A[2])*lT].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');}}}}

// ── BACKGROUND: SLOW NEBULA ───────────────────────────────────────────────────
function drawBg(){{
    bgX.fillStyle='rgba(0,0,0,0.15)';bgX.fillRect(0,0,W,H);
    const blobs=[
        {{bx:0.1+Math.sin(T*0.22)*0.15, by:0.15+Math.cos(T*0.18)*0.15, r:0.55, ci:0}},
        {{bx:0.9+Math.cos(T*0.19)*0.1,  by:0.85+Math.sin(T*0.25)*0.12, r:0.5,  ci:1}},
        {{bx:0.5+Math.sin(T*0.3+1)*0.25,by:0.05+Math.cos(T*0.22)*0.1,  r:0.4,  ci:2}},
        {{bx:0.2+Math.cos(T*0.17+2)*0.1,by:0.9+Math.sin(T*0.3)*0.08,   r:0.45, ci:3}},
        {{bx:mx/W, by:my/H, r:0.35, ci:1}},
    ];
    blobs.forEach(b=>{{
        const x=b.bx*W,y=b.by*H,r=b.r*Math.min(W,H);
        const [rr,gg,bb]=h2r(pal[b.ci]);
        const g=bgX.createRadialGradient(x,y,0,x,y,r);
        g.addColorStop(0,`rgba(${{rr}},${{gg}},${{bb}},0.18)`);
        g.addColorStop(0.6,`rgba(${{rr}},${{gg}},${{bb}},0.05)`);
        g.addColorStop(1,`rgba(${{rr}},${{gg}},${{bb}},0)`);
        bgX.fillStyle=g;bgX.fillRect(0,0,W,H);
    }});
}}

// ── STARS ─────────────────────────────────────────────────────────────────────
const STARS=Array.from({{length:200}},()=>{{
    const ci=Math.floor(Math.random()*4);
    return {{x:Math.random(),y:Math.random(),r:Math.random()*1.8+0.3,ci,phase:Math.random()*Math.PI*2,spd:0.5+Math.random()*2}};
}});
function drawStars(){{
    STARS.forEach(s=>{{
        const op=0.2+Math.sin(T*s.spd+s.phase)*0.4+0.4;
        const [rr,gg,bb]=h2r(pal[s.ci]);
        stX.beginPath();
        stX.arc(s.x*W,s.y*H,s.r*(0.8+Math.sin(T*s.spd+s.phase)*0.3),0,Math.PI*2);
        stX.fillStyle=`rgba(${{rr}},${{gg}},${{bb}},${{op}})`;
        stX.shadowBlur=s.r*6;stX.shadowColor=pal[s.ci];
        stX.fill();stX.shadowBlur=0;
    }});
}}

// ── NODE PIPES ────────────────────────────────────────────────────────────────
const NODES=Array.from({{length:10}},()=>{{
    const ci=Math.floor(Math.random()*4);
    return{{x:Math.random()*window.innerWidth,y:Math.random()*window.innerHeight,
            vx:(Math.random()-0.5)*0.7,vy:(Math.random()-0.5)*0.7,r:1.5+Math.random()*2,ci}};
}});
function drawPipes(){{
    NODES.forEach(n=>{{n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>W)n.vx*=-1;if(n.y<0||n.y>H)n.vy*=-1;}});
    for(let i=0;i<NODES.length;i++){{
        for(let j=i+1;j<NODES.length;j++){{
            const dx=NODES[j].x-NODES[i].x,dy=NODES[j].y-NODES[i].y,d=Math.sqrt(dx*dx+dy*dy);
            if(d<200){{
                const a=(1-d/200)*0.35,[rr,gg,bb]=h2r(pal[NODES[i].ci]);
                bgX.beginPath();bgX.moveTo(NODES[i].x,NODES[i].y);bgX.lineTo(NODES[j].x,NODES[j].y);
                bgX.strokeStyle=`rgba(${{rr}},${{gg}},${{bb}},${{a}})`;bgX.lineWidth=0.8;bgX.stroke();
            }}
        }}
        // Mouse
        const dx2=mx-NODES[i].x,dy2=my-NODES[i].y,d2=Math.sqrt(dx2*dx2+dy2*dy2);
        if(d2<160){{
            const a=(1-d2/160)*0.65,[rr,gg,bb]=h2r(pal[NODES[i].ci]);
            bgX.beginPath();bgX.moveTo(NODES[i].x,NODES[i].y);bgX.lineTo(mx,my);
            bgX.strokeStyle=`rgba(${{rr}},${{gg}},${{bb}},${{a}})`;bgX.lineWidth=1.2;bgX.stroke();
        }}
        const[rr,gg,bb]=h2r(pal[NODES[i].ci]);
        bgX.beginPath();bgX.arc(NODES[i].x,NODES[i].y,NODES[i].r,0,Math.PI*2);
        bgX.fillStyle=`rgba(${{rr}},${{gg}},${{bb}},0.8)`;
        bgX.shadowBlur=10;bgX.shadowColor=pal[NODES[i].ci];bgX.fill();bgX.shadowBlur=0;
    }}
}}

// ── PARTICLES ─────────────────────────────────────────────────────────────────
const DOTS=[];
class Dot{{
    constructor(x,y,boom=false){{
        this.x=x??Math.random()*W;
        this.y=y??H+5;
        this.vx=(Math.random()-0.5)*(boom?8:0.8);
        this.vy=boom?(Math.random()-0.5)*8-2:-(0.5+Math.random()*1.5);
        this.ci=Math.floor(Math.random()*4);
        this.r=boom?2+Math.random()*5:0.6+Math.random()*2.5;
        this.maxLife=boom?0.6+Math.random():4+Math.random()*7;
        this.age=0;this.angle=Math.random()*Math.PI*2;
        this.spin=(Math.random()-0.5)*0.1;
        this.shape=Math.floor(Math.random()*4);
        this.phase=Math.random()*Math.PI*2;
    }}
    update(){{
        this.x+=this.vx+(Math.random()-0.5)*0.12;
        this.y+=this.vy;this.vy-=0.005;
        this.age+=1/60;this.angle+=this.spin;
        const lr=this.age/this.maxLife;
        this.op=(lr<0.1?lr/0.1:lr>0.75?(1-lr)/0.25:1)*0.7*(0.7+Math.sin(T*3+this.phase)*0.3);
        return this.age<this.maxLife&&this.y>-20&&this.y<H+20&&this.x>-20&&this.x<W+20;
    }}
    draw(ctx){{
        const[rr,gg,bb]=h2r(pal[this.ci]);
        ctx.save();ctx.globalAlpha=Math.max(0,this.op);
        ctx.translate(this.x,this.y);ctx.rotate(this.angle);
        ctx.shadowBlur=this.r*5;ctx.shadowColor=pal[this.ci];
        const s=this.r,fill=`rgba(${{rr}},${{gg}},${{bb}},1)`;
        if(this.shape===0){{ctx.beginPath();ctx.arc(0,0,s,0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();}}
        else if(this.shape===1){{ctx.beginPath();ctx.moveTo(0,-s*1.5);ctx.lineTo(s*1.2,0);ctx.lineTo(0,s*1.5);ctx.lineTo(-s*1.2,0);ctx.closePath();ctx.fillStyle=fill;ctx.fill();}}
        else if(this.shape===2){{ctx.strokeStyle=fill;ctx.lineWidth=1.2;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(-s,0);ctx.lineTo(s,0);ctx.stroke();ctx.beginPath();ctx.moveTo(0,-s);ctx.lineTo(0,s);ctx.stroke();}}
        else{{ctx.beginPath();ctx.arc(0,0,s,0,Math.PI*2);ctx.strokeStyle=fill;ctx.lineWidth=1.2;ctx.stroke();}}
        ctx.restore();
    }}
}}

function burst(x,y,n){{for(let i=0;i<n;i++)DOTS.push(new Dot(x,y,true));}}
let lastMS=0;
function spawnMouse(){{const now=Date.now();if(now-lastMS>40&&DOTS.length<300){{DOTS.push(new Dot(mx,my,true));DOTS[DOTS.length-1].r*=0.6;lastMS=now;}}}}

// ── TITLE ANIMATION ───────────────────────────────────────────────────────────
const TITLE="DREAMWEAVER";
function animTitle(){{
    const el=document.getElementById('dwTitle');if(!el)return;
    el.innerHTML=TITLE.split('').map((ch,i)=>{{
        const ci=Math.floor(((T*0.5+i*0.18)%4+4)%4);
        const c=pal[ci];
        const dy=Math.sin(T*1.5+i*0.42)*6;
        const sc=0.94+Math.sin(T*2+i*0.5)*0.08;
        return`<span style="display:inline-block;color:${{c}};text-shadow:0 0 25px ${{c}},0 0 50px ${{c}}55;transform:translateY(${{dy.toFixed(1)}}px) scaleY(${{sc.toFixed(2)}});">${{ch}}</span>`;
    }}).join('');
    // Moon phase cycle
    const phases=['🌑','🌒','🌓','🌔','🌕','🌖','🌗','🌘'];
    const ph=document.getElementById('moonPhase');
    if(ph) ph.textContent=phases[Math.floor(T*0.3)%phases.length];
}}

// ── RENDER RESULT ─────────────────────────────────────────────────────────────
function renderResult(){{
    if(!RESULT) return;
    const r=RESULT;

    // Palette shift
    if(r.colors&&r.colors.length>=3){{
        tPal=[r.colors[0],r.colors[1]||r.colors[0],r.colors[2]||r.colors[0],r.colors[0]];
        lT=0;
    }}

    document.getElementById('idleState').style.display='none';
    document.getElementById('resultWrap').style.display='block';

    // Quote
    document.getElementById('cosmicQuote').textContent='"'+r.cosmic+'"';

    // Archetype card
    const c0=r.colors[0],c1=r.colors[1]||c0,c2=r.colors[2]||c0;
    const card=document.getElementById('archCard');
    card.style.borderColor=`rgba(${{h2r(c0).join(',')}},0.25)`;
    card.style.boxShadow=`0 0 100px ${{c0}}33, 0 0 200px ${{c0}}11`;
    document.getElementById('archBg').style.background=`linear-gradient(135deg,${{r.colors.join(',')}})`;
    document.getElementById('archGlow').style.background=c0;

    const glyph=document.getElementById('archGlyph');
    glyph.textContent=r.glyph;
    glyph.style.color=c0;
    glyph.style.textShadow=`0 0 30px ${{c0}},0 0 60px ${{c1}}`;

    const name=document.getElementById('archName');
    name.textContent=r.primary;
    name.style.background=`linear-gradient(135deg,${{r.colors.join(',')}})`;
    name.style.webkitBackgroundClip='text';
    name.style.webkitTextFillColor='transparent';
    name.style.backgroundClip='text';

    [['tagElement','⬡ '+r.element],['tagPlanet','◎ '+r.planet],['tagNumber','# '+r.number]].forEach(([id,txt])=>{{
        const el=document.getElementById(id);
        el.textContent=txt;el.style.color=c1;el.style.borderColor=c1+'55';el.style.fontSize='0.52rem';el.style.letterSpacing='0.18em';
    }});

    document.getElementById('prophecyText').textContent=r.prophecy;
    document.getElementById('shadowText').textContent=r.shadow;

    // Shadow box border
    const shadowBox=document.querySelector('#archCard [style*="border-left:3px"]');
    if(shadowBox) shadowBox.style.borderLeftColor=c0+'88';

    document.getElementById('ritualText').textContent=r.ritual;

    // Symbols
    if(r.symbols&&r.symbols.length>0){{
        document.getElementById('symbolsWrap').style.display='block';
        const grid=document.getElementById('symbolsGrid');
        r.symbols.forEach(([sym,meaning,col],i)=>{{
            const card=document.createElement('div');
            card.style.cssText=`background:rgba(10,0,30,0.6);border:1px solid ${{col}}33;border-radius:14px;padding:14px 16px;animation:fadeUp 0.5s ease ${{0.5+i*0.1}}s both;opacity:0;backdrop-filter:blur(20px);`;
            card.innerHTML=`<div style="font-size:0.55rem;letter-spacing:0.2em;color:${{col}};text-transform:uppercase;font-family:monospace;font-weight:700;margin-bottom:6px;">${{sym.toUpperCase()}}</div><div style="font-size:0.78rem;color:rgba(200,185,255,0.7);line-height:1.5;font-style:italic;font-family:Georgia,serif;">${{meaning}}</div>`;
            grid.appendChild(card);
        }});
    }}

    // Intensity
    document.getElementById('intensityBar').style.cssText+=`width:${{r.intensity}}%;background:linear-gradient(90deg,${{r.colors.join(',')}});box-shadow:0 0 20px ${{c0}};`;
    document.getElementById('intensityPct').textContent=r.intensity+'% · '+r.word_count+' words';

    if(r.secondary){{
        const sl=document.getElementById('secondaryLine');
        sl.style.display='block';
        sl.textContent='✦ Secondary resonance: '+r.secondary;
    }}

    // Big burst
    burst(W/2,H/2,60);
    setTimeout(()=>burst(W*0.3,H*0.3,20),300);
    setTimeout(()=>burst(W*0.7,H*0.7,20),600);
}}

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────
function loop(){{
    T+=0.016;stepPal();drawBg();drawPipes();
    stX.clearRect(0,0,W,H);drawStars();
    if(Math.random()<0.4&&DOTS.length<250) DOTS.push(new Dot());
    for(let i=DOTS.length-1;i>=0;i--){{if(!DOTS[i].update())DOTS.splice(i,1);else DOTS[i].draw(stX);}}
    animTitle();
    requestAnimationFrame(loop);
}}

loop();
if(RESULT) setTimeout(renderResult,200);

// Warn toast auto-hide
const wt=document.getElementById('warnToast');
if({str(warn).lower()}){{
    wt.style.transform='translateX(-50%) translateY(0)';
    setTimeout(()=>{{wt.style.transform='translateX(-50%) translateY(80px)';}},2800);
}}

}})();
</script>
"""

# ── APP LOGIC ─────────────────────────────────────────────────────────────────
def main():
    result = st.session_state.get("result")
    warn   = st.session_state.pop("warn", False)

    # Render full canvas page
    st.markdown(build_page(result=result, warn=warn), unsafe_allow_html=True)

    # Functional Streamlit form (visually overlaid by canvas UI)
    with st.form("dream_form", clear_on_submit=False):
        text = st.text_area("dream", key="dream_text",
            placeholder="Describe your dream… fragments, feelings, images, whatever you remember…",
            height=1, label_visibility="collapsed")
        go = st.form_submit_button("🌑  DECODE THE DREAM  🌑")

    if go:
        raw = st.session_state.get("dream_text","").strip()
        if raw:
            with st.spinner(""):
                r = decode_dream(raw)
            st.session_state["result"] = r
            st.rerun()
        else:
            st.session_state["warn"] = True
            st.rerun()

if __name__ == "__main__":
    main()

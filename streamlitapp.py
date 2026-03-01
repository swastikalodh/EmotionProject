"""
🔮 SOUL PRISM — Emotion Alchemy Engine v3
Completely new design: canvas particles, pipe connections, dreamy palette shifts
Requirements: pip install streamlit nltk deep-translator
Run: streamlit run vibe_oracle.py
"""

import streamlit as st
import re
import json

st.set_page_config(
    page_title="🔮 Soul Prism",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── NLP ───────────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    for pkg, path in [("stopwords","corpora/stopwords"),("wordnet","corpora/wordnet"),
                      ("vader_lexicon","sentiment/vader_lexicon"),("punkt","tokenizers/punkt")]:
        try: nltk.data.find(path)
        except LookupError: nltk.download(pkg, quiet=True)
    _L = WordNetLemmatizer(); _SW = set(stopwords.words("english"))
    _SIA = SentimentIntensityAnalyzer(); NLP_OK = True
except: NLP_OK = False

EMOTIONS = {
    "joy":     {"colors":["#FFE100","#FF8C00","#FF3CAC","#784BA0"],"label":"JOY",     "sym":"☀","desc":"Radiant. Electric. Alive."},
    "anger":   {"colors":["#FF0040","#FF4500","#FF9500","#FFE000"],"label":"RAGE",    "sym":"⚡","desc":"Volcanic. Fierce. Unstoppable."},
    "sadness": {"colors":["#0055FF","#0099FF","#00E5FF","#A0F0FF"],"label":"SORROW",  "sym":"◎","desc":"Deep. Vast. Oceanic."},
    "fear":    {"colors":["#6600FF","#AA00FF","#FF00CC","#FF0055"],"label":"DREAD",   "sym":"◉","desc":"Shadowed. Trembling. Raw."},
    "disgust": {"colors":["#00FF66","#00FFCC","#00CCFF","#0066FF"],"label":"REVULSION","sym":"❋","desc":"Sharp. Knowing. Untamed."},
    "surprise":{"colors":["#FF00FF","#FF0099","#FF6600","#FFFF00"],"label":"WONDER",  "sym":"★","desc":"Impossible. Blazing. Infinite."},
}

KEYWORDS = {
    "joy":["happy","happiness","joy","love","elated","cheerful","excited","wonderful","amazing","great","awesome","bliss","delight","thrilled","ecstatic","glad","smile","celebrate","fun","enjoy","grateful","euphoric","radiant","gleeful","overjoyed","beautiful","divine","bright","warm","peaceful","blessed","lucky","free","alive","glow","light","hope","wonderful","fantastic"],
    "anger":["angry","anger","furious","rage","mad","irritated","annoyed","frustrated","hate","outraged","livid","enraged","hostile","bitter","resentful","infuriated","irate","seething","fuming","explosive","violent","scream","yell","fed","unfair","disgusting","terrible","horrible","worst","awful","sick","cannot","unacceptable"],
    "sadness":["sad","unhappy","depressed","grief","sorrow","cry","crying","lonely","heartbroken","miserable","gloomy","melancholy","hopeless","devastated","despair","broken","lost","empty","numb","miss","alone","dark","pain","hurt","weep","abandoned","forgotten","invisible","tears","mourn","ache"],
    "fear":["afraid","fear","scared","frightened","terrified","anxious","panic","worried","nervous","dread","horror","terror","uneasy","paranoid","trembling","horrified","petrified","alarmed","nightmare","danger","trapped","helpless","overwhelm","shaking","tense","threat"],
    "disgust":["disgusted","disgust","gross","revolting","repulsed","nauseated","sick","nasty","vile","loathe","offensive","foul","rotten","toxic","unbearable","hideous","awful","dreadful","corrupt","yuck","eww"],
    "surprise":["surprised","shocked","astonished","amazed","stunned","startled","unexpected","unbelievable","wow","whoa","incredible","astounded","speechless","bewildered","dazzled","impossible","never","omg","wait","unreal","mindblowing","extraordinary","unbelievable","cannot believe"],
}

MULTILINGUAL = {
    "खुश":"joy","खुशी":"joy","प्यार":"joy","आनंद":"joy","गुस्सा":"anger","क्रोध":"anger",
    "दुख":"sadness","उदास":"sadness","डर":"fear","भय":"fear","घृणा":"disgust","अचरज":"surprise",
    "আনন্দ":"joy","খুশি":"joy","রাগ":"anger","দুঃখ":"sadness","ভয়":"fear","ঘৃণা":"disgust","আশ্চর্য":"surprise",
}

def detect(text):
    scores = {e:0 for e in EMOTIONS}
    for phrase, em in MULTILINGUAL.items():
        if phrase in text: scores[em] += 2
    raw = text.lower()
    raw = re.sub(r"[^\w\s]"," ", raw)
    tokens = raw.split()
    if NLP_OK:
        tokens = [_L.lemmatize(w) for w in tokens if w not in _SW]
    for em, kws in KEYWORDS.items():
        for tok in tokens:
            if tok in kws: scores[em] += 1
    total = sum(scores.values())
    if total == 0:
        comp = _SIA.polarity_scores(text)["compound"] if NLP_OK else 0
        if   comp >= 0.05:  scores["joy"]     = 1
        elif comp <= -0.05: scores["sadness"]  = 1
        else:               scores["surprise"] = 1
        total = sum(scores.values())
    if total == 0: scores["surprise"] = 1; total = 1
    return {e: round(v/total*100,1) for e,v in scores.items()}

def dominant(probs): return max(probs, key=probs.get)


# ── RENDER ────────────────────────────────────────────────────────────────────
def main():
    probs = st.session_state.get("probs")
    dom   = st.session_state.get("dom")

    # Build result JSON for JS
    result_json = "null"
    if probs and dom:
        ed = EMOTIONS[dom]
        spectrum = []
        for e, pct in sorted(probs.items(), key=lambda x:-x[1]):
            ed2 = EMOTIONS[e]
            spectrum.append({"label":ed2["label"],"sym":ed2["sym"],"pct":pct,"colors":ed2["colors"]})
        result_json = json.dumps({
            "dom": dom,
            "label": ed["label"],
            "sym": ed["sym"],
            "desc": ed["desc"],
            "colors": ed["colors"],
            "spectrum": spectrum,
        })

    # Streamlit form (will be styled/hidden by CSS, but functional)
    with st.form("soul_form", clear_on_submit=False):
        text = st.text_area("soul", key="soul_text",
            placeholder="Pour your soul here… the stars are listening",
            height=1, label_visibility="collapsed")
        submitted = st.form_submit_button("DECODE")

    if submitted:
        raw = st.session_state.get("soul_text","").strip()
        if raw:
            with st.spinner(""):
                p = detect(raw)
            d = dominant(p)
            st.session_state["probs"] = p
            st.session_state["dom"]   = d
            st.rerun()
        else:
            st.session_state["show_warn"] = True

    show_warn = st.session_state.pop("show_warn", False)

    # ── THE ENTIRE VISUAL EXPERIENCE ──
    st.markdown(build_html(result_json, show_warn), unsafe_allow_html=True)


def build_html(result_json: str, show_warn: bool) -> str:
    warn_class = "show" if show_warn else ""
    return f"""
<style>
/* ── RESET & HIDE STREAMLIT ── */
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stDeployButton"],.stDeployButton,[data-testid="stStatusWidget"]
{{ display:none!important; }}
[data-testid="stAppViewContainer"]>.main {{ padding:0!important; }}
section.main>.block-container {{ padding:0!important; max-width:100%!important; }}
html,body,[data-testid="stAppViewContainer"] {{
    background:#000!important; overflow:hidden!important;
    height:100vh!important; width:100vw!important;
}}
[data-testid="stVerticalBlock"] {{ gap:0!important; }}

/* ── HIDE STREAMLIT FORM VISUALLY BUT KEEP FUNCTIONAL ── */
[data-testid="stForm"] {{
    position:fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 99999 !important;
    width: min(720px, 94vw) !important;
    background: transparent !important;
    border: none !important;
    padding: 0 0 32px 0 !important;
    pointer-events: auto !important;
}}
[data-testid="stTextArea"] {{
    position:fixed !important;
    bottom:130px !important;
    left:50% !important;
    transform:translateX(-50%) !important;
    z-index:99999 !important;
    width:min(720px,94vw) !important;
}}
[data-testid="stTextArea"] textarea {{
    background: rgba(8,0,20,0.82) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 20px !important;
    color: #f0e8ff !important;
    font-family: Georgia, serif !important;
    font-style: italic !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    padding: 18px 22px !important;
    resize: none !important;
    outline: none !important;
    backdrop-filter: blur(30px) !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 0 40px rgba(120,0,255,0.15) !important;
    height: 100px !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border-color: rgba(255,255,255,0.4) !important;
    box-shadow: 0 0 60px rgba(180,0,255,0.35), 0 0 120px rgba(180,0,255,0.15) !important;
}}
[data-testid="stTextArea"] textarea::placeholder {{
    color: rgba(200,160,255,0.35) !important;
}}
[data-testid="stTextArea"] label {{ display:none !important; }}
[data-testid="stFormSubmitButton"] {{
    position: fixed !important;
    bottom: 36px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 99999 !important;
    width: min(720px,94vw) !important;
}}
[data-testid="stFormSubmitButton"] button {{
    width: 100% !important;
    padding: 17px 44px !important;
    border: none !important;
    border-radius: 60px !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    background: linear-gradient(135deg,#6600ff,#ff00aa,#ff6600,#ffe100) !important;
    background-size: 400% 400% !important;
    animation: btnFlow 3s ease infinite !important;
    color: #fff !important;
    box-shadow: 0 0 50px rgba(120,0,255,0.5) !important;
    transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.25s !important;
}}
[data-testid="stFormSubmitButton"] button:hover {{
    transform: scale(1.05) translateY(-2px) !important;
    box-shadow: 0 0 80px rgba(180,0,255,0.75) !important;
}}
[data-testid="stSpinner"] {{ display:none!important; }}

@keyframes btnFlow {{
    0%  {{ background-position:0% 50%; }}
    50% {{ background-position:100% 50%; }}
    100%{{ background-position:0% 50%; }}
}}
</style>

<!-- FULL-SCREEN CANVAS EXPERIENCE -->
<div id="SP" style="position:fixed;inset:0;width:100vw;height:100vh;overflow:hidden;z-index:100;pointer-events:none;">
  <canvas id="bgC" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
  <canvas id="ptC" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>

  <!-- UI OVERLAY -->
  <div id="uiLayer" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;padding-top:clamp(24px,5vh,60px);">

    <!-- TITLE -->
    <div id="titleWrap" style="text-align:center;margin-bottom:clamp(12px,3vh,32px);">
      <div id="titleText" style="font-size:clamp(2.2rem,6vw,4.5rem);font-weight:900;letter-spacing:0.06em;cursor:default;user-select:none;line-height:1;"></div>
      <div style="font-size:clamp(0.55rem,1.1vw,0.75rem);letter-spacing:0.45em;color:rgba(255,255,255,0.3);margin-top:10px;text-transform:uppercase;font-family:monospace;">Emotion Alchemy Engine</div>
      <div id="runeBar" style="margin-top:12px;font-size:1.1rem;letter-spacing:0.6em;color:rgba(255,255,255,0.15);animation:runeGlow 4s ease-in-out infinite;">ᚷ ᛃ ᛇ ᚾ ᚦ ᛚ</div>
    </div>

    <!-- RESULT PANEL -->
    <div id="resultPanel" style="display:none;width:min(700px,92vw);animation:panelIn 0.9s cubic-bezier(0.34,1.56,0.64,1) both;">

      <!-- EMOTION HERO -->
      <div id="heroBlock" style="
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:28px;
        padding:clamp(18px,3vw,32px) clamp(20px,4vw,40px);
        margin-bottom:14px;
        display:flex;align-items:center;gap:24px;
        position:relative;overflow:hidden;
        backdrop-filter:blur(24px);
      ">
        <div id="heroBg" style="position:absolute;inset:0;border-radius:28px;opacity:0.07;pointer-events:none;"></div>
        <div id="heroSym" style="font-size:clamp(3rem,7vw,5.5rem);line-height:1;animation:heroSpin 5s ease-in-out infinite;position:relative;z-index:1;filter:drop-shadow(0 0 30px currentColor);"></div>
        <div style="flex:1;position:relative;z-index:1;">
          <div id="heroLabel" style="font-size:clamp(1.8rem,4.5vw,3.2rem);font-weight:900;letter-spacing:0.2em;line-height:1;margin-bottom:8px;"></div>
          <div id="heroDesc" style="font-size:clamp(0.75rem,1.4vw,0.95rem);letter-spacing:0.14em;opacity:0.6;font-style:italic;font-family:Georgia,serif;"></div>
        </div>
        <!-- PIPE BARS right side -->
        <div id="pipeBars" style="display:flex;align-items:flex-end;gap:6px;height:60px;position:relative;z-index:1;"></div>
      </div>

      <!-- SPECTRUM -->
      <div id="spectrumWrap" style="
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.06);
        border-radius:22px;
        padding:clamp(14px,2.5vw,24px) clamp(16px,3vw,28px);
        backdrop-filter:blur(20px);
      ">
        <div style="font-size:0.55rem;letter-spacing:0.3em;color:rgba(255,255,255,0.25);text-transform:uppercase;text-align:center;margin-bottom:14px;font-family:monospace;">◈ &nbsp; Full Emotional Spectrum &nbsp; ◈</div>
        <div id="specRows"></div>
      </div>

    </div>

    <!-- SPACER so result doesn't overlap inputs -->
    <div style="flex:1;min-height:20px;"></div>
    <!-- Extra bottom space for fixed input -->
    <div style="height:200px;"></div>
  </div>

  <!-- WARNING TOAST -->
  <div id="warnToast" class="{warn_class}" style="
    position:fixed;bottom:32px;left:50%;
    transform:translateX(-50%) translateY(120px);
    background:rgba(255,100,0,0.15);
    border:1px solid rgba(255,160,50,0.45);
    border-radius:60px;padding:13px 30px;
    color:#ffcc88;font-size:0.78rem;letter-spacing:0.15em;
    backdrop-filter:blur(20px);
    transition:transform 0.5s cubic-bezier(0.34,1.56,0.64,1);
    white-space:nowrap;z-index:200000;pointer-events:none;
  ">⚠ &nbsp; Pour something into the void first &nbsp; ⚠</div>

</div>

<style>
@keyframes runeGlow {{
    0%,100%{{ opacity:0.1; }}
    50%    {{ opacity:0.45; text-shadow:0 0 20px rgba(160,100,255,0.8); }}
}}
@keyframes panelIn {{
    from{{ opacity:0; transform:translateY(30px) scale(0.95); }}
    to  {{ opacity:1; transform:translateY(0) scale(1); }}
}}
@keyframes heroSpin {{
    0%,100%{{ transform:scale(1) rotate(-5deg); }}
    33%    {{ transform:scale(1.12) rotate(5deg); }}
    66%    {{ transform:scale(1.06) rotate(-3deg); }}
}}
@keyframes specRowIn {{
    from{{ opacity:0; transform:translateX(-20px); }}
    to  {{ opacity:1; transform:translateX(0); }}
}}
@keyframes pipeFill {{
    from{{ width:0!important; }}
}}
@keyframes pipeShine {{
    0%  {{ left:-80%; }}
    100%{{ left:180%; }}
}}
@keyframes warnShow {{
    to{{ transform:translateX(-50%) translateY(0); }}
}}
#warnToast.show {{
    animation: warnShow 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards,
               warnHide 0.5s ease 2.5s forwards;
}}
@keyframes warnHide {{
    to{{ transform:translateX(-50%) translateY(120px); }}
}}
@keyframes titleLetterFloat {{
    0%,100%{{ transform:translateY(0) scaleY(1); }}
    50%    {{ transform:translateY(-6px) scaleY(1.05); }}
}}
</style>

<script>
(function() {{
const RESULT = {result_json};

// ── CANVAS ────────────────────────────────────────────────────────────────────
const bgC = document.getElementById('bgC');
const ptC = document.getElementById('ptC');
const bgX = bgC.getContext('2d');
const ptX = ptC.getContext('2d');
let W, H, t=0, mx=0, my=0;

// Dynamic color palette
const DEFAULT_PAL = ['#6600FF','#FF00AA','#00CCFF','#FFD700'];
let pal = DEFAULT_PAL.slice();
let targetPal = pal.slice();

function resize() {{
    W = bgC.width = ptC.width = window.innerWidth;
    H = bgC.height= ptC.height= window.innerHeight;
}}
resize();
window.addEventListener('resize', resize);

document.addEventListener('mousemove', e=>{{ mx=e.clientX; my=e.clientY; spawnMouse(); }});
document.addEventListener('touchmove', e=>{{ mx=e.touches[0].clientX; my=e.touches[0].clientY; }},{{passive:true}});
document.addEventListener('click', e=>{{
    for(let i=0;i<20;i++) {{
        const p=new Dot(true);
        p.x=e.clientX; p.y=e.clientY;
        p.vx=(Math.random()-0.5)*7;
        p.vy=(Math.random()-0.5)*7-2;
        DOTS.push(p);
    }}
}});

// ── COLOR LERP ────────────────────────────────────────────────────────────────
function h2r(h) {{
    h=h.replace('#','');
    if(h.length===3) h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
    return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];
}}
function r2h(r,g,b) {{
    return '#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');
}}

let lerpT = 0;
function stepPal() {{
    lerpT = Math.min(lerpT + 0.012, 1);
    for(let i=0;i<4;i++) {{
        const A=h2r(pal[i]), B=h2r(targetPal[i]);
        pal[i] = r2h(A[0]+(B[0]-A[0])*lerpT, A[1]+(B[1]-A[1])*lerpT, A[2]+(B[2]-A[2])*lerpT);
    }}
}}

// ── FLUID BLOB BG ─────────────────────────────────────────────────────────────
function drawBg() {{
    bgX.fillStyle = 'rgba(0,0,0,0.14)';
    bgX.fillRect(0,0,W,H);

    const blobs = [
        {{bx:0.12+Math.sin(t*0.38)*0.14, by:0.18+Math.cos(t*0.29)*0.14, br:0.42, ci:0}},
        {{bx:0.88+Math.cos(t*0.32)*0.1,  by:0.78+Math.sin(t*0.42)*0.12, br:0.38, ci:1}},
        {{bx:0.55+Math.sin(t*0.48+1)*0.2,by:0.08+Math.cos(t*0.38+2)*0.1,br:0.3,  ci:2}},
        {{bx:0.28+Math.cos(t*0.28+3)*0.14,by:0.82+Math.sin(t*0.48)*0.1, br:0.34, ci:3}},
        {{bx:0.72+Math.sin(t*0.42+2)*0.12,by:0.42+Math.cos(t*0.32)*0.2, br:0.3,  ci:0}},
        {{bx:mx/W, by:my/H, br:0.28, ci:1}},  // follows mouse
    ];

    blobs.forEach(b => {{
        const x=b.bx*W, y=b.by*H, r=b.br*Math.min(W,H);
        const [rr,gg,bb] = h2r(pal[b.ci]);
        const g = bgX.createRadialGradient(x,y,0,x,y,r);
        g.addColorStop(0,  `rgba(${{rr}},${{gg}},${{bb}},0.22)`);
        g.addColorStop(0.5,`rgba(${{rr}},${{gg}},${{bb}},0.08)`);
        g.addColorStop(1,  `rgba(${{rr}},${{gg}},${{bb}},0)`);
        bgX.fillStyle=g;
        bgX.fillRect(0,0,W,H);
    }});
}}

// ── NODES & PIPES (connective lines) ──────────────────────────────────────────
const NODES = Array.from({{length:14}},()=>{{
    const ci=Math.floor(Math.random()*4);
    return {{
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        vx:(Math.random()-0.5)*0.9,
        vy:(Math.random()-0.5)*0.9,
        r:1.5+Math.random()*2.5,
        ci,
    }};
}});

function drawNodes() {{
    NODES.forEach(n => {{
        n.x+=n.vx; n.y+=n.vy;
        if(n.x<0||n.x>W) n.vx*=-1;
        if(n.y<0||n.y>H) n.vy*=-1;
    }});
    for(let i=0;i<NODES.length;i++) {{
        for(let j=i+1;j<NODES.length;j++) {{
            const dx=NODES[j].x-NODES[i].x,dy=NODES[j].y-NODES[i].y;
            const d=Math.sqrt(dx*dx+dy*dy);
            if(d<220){{
                const a=(1-d/220)*0.4;
                const [rr,gg,bb]=h2r(pal[NODES[i].ci]);
                bgX.beginPath();
                bgX.moveTo(NODES[i].x,NODES[i].y);
                bgX.lineTo(NODES[j].x,NODES[j].y);
                bgX.strokeStyle=`rgba(${{rr}},${{gg}},${{bb}},${{a}})`;
                bgX.lineWidth=1;
                bgX.stroke();
            }}
        }}
        // Mouse connection
        const dx2=mx-NODES[i].x, dy2=my-NODES[i].y;
        const d2=Math.sqrt(dx2*dx2+dy2*dy2);
        if(d2<180){{
            const a=(1-d2/180)*0.7;
            const [rr,gg,bb]=h2r(pal[NODES[i].ci]);
            bgX.beginPath();
            bgX.moveTo(NODES[i].x,NODES[i].y);
            bgX.lineTo(mx,my);
            bgX.strokeStyle=`rgba(${{rr}},${{gg}},${{bb}},${{a}})`;
            bgX.lineWidth=1.5;
            bgX.stroke();
        }}
        // Node dot
        const [rr,gg,bb]=h2r(pal[NODES[i].ci]);
        bgX.beginPath();
        bgX.arc(NODES[i].x,NODES[i].y,NODES[i].r,0,Math.PI*2);
        bgX.fillStyle=`rgba(${{rr}},${{gg}},${{bb}},0.85)`;
        bgX.shadowBlur=12; bgX.shadowColor=pal[NODES[i].ci];
        bgX.fill();
        bgX.shadowBlur=0;
    }}
}}

// ── PARTICLES ─────────────────────────────────────────────────────────────────
const DOTS = [];
const SHAPES = ['circle','diamond','cross','ring','star'];

class Dot {{
    constructor(fromMouse=false) {{
        this.fromMouse = fromMouse;
        if(fromMouse) {{
            this.x=mx+(Math.random()-0.5)*16;
            this.y=my+(Math.random()-0.5)*16;
            this.vx=(Math.random()-0.5)*4;
            this.vy=(Math.random()-0.5)*4-1.5;
            this.maxLife=0.5+Math.random()*0.8;
        }} else {{
            this.x=Math.random()*W;
            this.y=H+10;
            this.vx=(Math.random()-0.5)*0.7;
            this.vy=-(0.4+Math.random()*1.4);
            this.maxLife=5+Math.random()*8;
        }}
        this.ci=Math.floor(Math.random()*4);
        this.r=fromMouse ? 2+Math.random()*5 : 0.8+Math.random()*3;
        this.age=0;
        this.angle=Math.random()*Math.PI*2;
        this.spin=(Math.random()-0.5)*0.08;
        this.shape=SHAPES[Math.floor(Math.random()*SHAPES.length)];
        this.twinkle=Math.random()*Math.PI*2;
    }}
    update() {{
        this.x+=this.vx+(Math.random()-0.5)*0.15;
        this.y+=this.vy;
        this.vy-=0.006;
        this.age+=1/60;
        this.angle+=this.spin;
        this.twinkle+=0.08;
        const lr=this.age/this.maxLife;
        this.op=this.fromMouse
            ? Math.sin(lr*Math.PI)*0.9
            : (lr<0.08?lr/0.08:lr>0.75?(1-lr)/0.25:1)*0.75;
        return this.age<this.maxLife&&this.x>-10&&this.x<W+10&&this.y>-10;
    }}
    draw(ctx) {{
        const [rr,gg,bb]=h2r(pal[this.ci]);
        const twinkleOp = this.op * (0.7+Math.sin(this.twinkle)*0.3);
        ctx.save();
        ctx.globalAlpha=twinkleOp;
        ctx.translate(this.x,this.y);
        ctx.rotate(this.angle);
        ctx.shadowBlur=this.r*5;
        ctx.shadowColor=pal[this.ci];
        const fill=`rgba(${{rr}},${{gg}},${{bb}},1)`;
        const stroke=`rgba(${{rr}},${{gg}},${{bb}},0.8)`;
        const s=this.r;
        if(this.shape==='circle'){{
            ctx.beginPath();ctx.arc(0,0,s,0,Math.PI*2);
            ctx.fillStyle=fill;ctx.fill();
        }} else if(this.shape==='diamond'){{
            ctx.beginPath();ctx.moveTo(0,-s*1.5);ctx.lineTo(s*1.2,0);ctx.lineTo(0,s*1.5);ctx.lineTo(-s*1.2,0);ctx.closePath();
            ctx.fillStyle=fill;ctx.fill();
        }} else if(this.shape==='cross'){{
            ctx.strokeStyle=stroke;ctx.lineWidth=s*0.6;ctx.lineCap='round';
            ctx.beginPath();ctx.moveTo(-s,0);ctx.lineTo(s,0);ctx.stroke();
            ctx.beginPath();ctx.moveTo(0,-s);ctx.lineTo(0,s);ctx.stroke();
        }} else if(this.shape==='ring'){{
            ctx.beginPath();ctx.arc(0,0,s,0,Math.PI*2);
            ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();
        }} else {{  // star
            ctx.fillStyle=fill;
            for(let i=0;i<5;i++){{
                const a=i*Math.PI*2/5-Math.PI/2;
                const b=a+Math.PI/5;
                if(i===0) ctx.beginPath();
                ctx.lineTo(Math.cos(a)*s,Math.sin(a)*s);
                ctx.lineTo(Math.cos(b)*s*0.45,Math.sin(b)*s*0.45);
            }}
            ctx.closePath();ctx.fill();
        }}
        ctx.restore();
    }}
}}

let lastMSpawn=0;
function spawnMouse(){{
    const now=Date.now();
    if(now-lastMSpawn>35&&DOTS.length<250){{
        DOTS.push(new Dot(true));DOTS.push(new Dot(true));
        lastMSpawn=now;
    }}
}}

function updateDots(){{
    for(let i=DOTS.length-1;i>=0;i--){{
        if(!DOTS[i].update()) DOTS.splice(i,1);
    }}
}}
function drawDots(){{
    ptX.clearRect(0,0,W,H);
    DOTS.forEach(d=>d.draw(ptX));
}}

// ── TITLE ─────────────────────────────────────────────────────────────────────
const TITLE_STR = "SOUL PRISM";
function animTitle(){{
    const el=document.getElementById('titleText');
    if(!el) return;
    el.innerHTML=TITLE_STR.split('').map((ch,i)=>{{
        if(ch===' ') return `<span>&nbsp;</span>`;
        const ci=Math.floor(((t*0.6+i*0.2)%4+4)%4);
        const c=pal[ci];
        const dy=Math.sin(t*1.8+i*0.45)*5;
        const sy=0.94+Math.sin(t*2.2+i*0.5)*0.08;
        return `<span style="display:inline-block;color:${{c}};text-shadow:0 0 30px ${{c}},0 0 60px ${{c}}66;transform:translateY(${{dy.toFixed(1)}}px) scaleY(${{sy.toFixed(2)}});transition:color 0.6s;">${{ch}}</span>`;
    }}).join('');
}}

// ── RESULT RENDERING ──────────────────────────────────────────────────────────
function renderResult(){{
    if(!RESULT) return;
    const r=RESULT;
    targetPal=r.colors.slice();lerpT=0;

    document.getElementById('resultPanel').style.display='block';

    // Hero
    const hero=document.getElementById('heroBlock');
    const heroBg=document.getElementById('heroBg');
    const heroSym=document.getElementById('heroSym');
    const heroLabel=document.getElementById('heroLabel');
    const heroDesc=document.getElementById('heroDesc');

    heroBg.style.background=`linear-gradient(135deg,${{r.colors.join(',')}})`;
    hero.style.boxShadow=`0 0 80px ${{r.colors[0]}}44,0 0 160px ${{r.colors[0]}}22`;
    hero.style.borderColor=`rgba(${{h2r(r.colors[0]).join(',')}},0.3)`;

    heroSym.textContent=r.sym;
    heroSym.style.color=r.colors[0];
    heroSym.style.textShadow=`0 0 40px ${{r.colors[0]}},0 0 80px ${{r.colors[1]}}`;

    heroLabel.textContent=r.label;
    heroLabel.style.background=`linear-gradient(135deg,${{r.colors.join(',')}})`;
    heroLabel.style.webkitBackgroundClip='text';
    heroLabel.style.webkitTextFillColor='transparent';
    heroLabel.style.backgroundClip='text';

    heroDesc.textContent=r.desc;
    heroDesc.style.color=r.colors[2];

    // Pipe bars (animated vertical bars)
    const pb=document.getElementById('pipeBars');
    pb.innerHTML='';
    r.colors.forEach((c,i)=>{{
        const heights=[28,44,36,52];
        const bar=document.createElement('div');
        bar.style.cssText=`width:6px;height:${{heights[i]}}px;border-radius:6px;background:${{c}};box-shadow:0 0 16px ${{c}};animation:heroBarGrow 0.8s cubic-bezier(0.34,1.56,0.64,1) ${{i*0.12}}s both;`;
        pb.appendChild(bar);
    }});

    // Spectrum
    const sr=document.getElementById('specRows');
    sr.innerHTML='';
    r.spectrum.forEach((em,idx)=>{{
        const row=document.createElement('div');
        row.style.cssText=`display:flex;align-items:center;gap:14px;margin-bottom:10px;animation:specRowIn 0.5s ease ${{0.1+idx*0.08}}s both;opacity:0;`;

        const lbl=document.createElement('div');
        lbl.style.cssText=`font-size:0.56rem;letter-spacing:0.12em;min-width:74px;text-align:right;text-transform:uppercase;color:${{em.colors[0]}};font-weight:700;font-family:monospace;line-height:1.3;`;
        lbl.innerHTML=`${{em.sym}}<br>${{em.label}}`;

        const track=document.createElement('div');
        track.style.cssText=`flex:1;height:11px;background:rgba(255,255,255,0.05);border-radius:11px;overflow:hidden;position:relative;border:1px solid rgba(255,255,255,0.06);`;

        const fill=document.createElement('div');
        fill.style.cssText=`height:100%;width:${{Math.max(em.pct,1)}}%;border-radius:11px;background:linear-gradient(90deg,${{em.colors.join(',')}});box-shadow:0 0 14px ${{em.colors[0]}};animation:pipeFill 1.3s cubic-bezier(0.34,1.56,0.64,1) ${{0.2+idx*0.08}}s both;position:relative;overflow:hidden;`;

        const shine=document.createElement('div');
        shine.style.cssText=`position:absolute;top:0;left:-80%;width:60%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.55),transparent);animation:pipeShine 2.2s ease-in-out ${{idx*0.15}}s infinite;`;
        fill.appendChild(shine);
        track.appendChild(fill);

        const pct=document.createElement('div');
        pct.style.cssText=`font-size:0.62rem;min-width:34px;color:rgba(255,255,255,0.38);font-family:monospace;`;
        pct.textContent=em.pct+'%';

        row.appendChild(lbl);row.appendChild(track);row.appendChild(pct);
        sr.appendChild(row);
    }});

    // Burst of particles
    for(let i=0;i<80;i++) setTimeout(()=>DOTS.push(new Dot(true)), i*15);
}}

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────
function loop(){{
    t+=0.016;
    stepPal();
    drawBg();
    drawNodes();
    if(Math.random()<0.35&&DOTS.length<220) DOTS.push(new Dot(false));
    updateDots();
    drawDots();
    animTitle();
    requestAnimationFrame(loop);
}}

// ── INIT ──────────────────────────────────────────────────────────────────────
loop();
if(RESULT) setTimeout(renderResult, 150);

// Textarea dynamic glow
document.addEventListener('DOMContentLoaded',()=>{{
    const obs=new MutationObserver(()=>{{
        const ta=document.querySelector('[data-testid="stTextArea"] textarea');
        if(ta&&!ta.__enhanced) {{
            ta.__enhanced=true;
            ta.addEventListener('input',()=>{{
                for(let i=0;i<4;i++) DOTS.push(new Dot(true));
            }});
        }}
    }});
    obs.observe(document.body,{{childList:true,subtree:true}});
}});

}})();
</script>

<style>
@keyframes heroBarGrow {{
    from{{ height:0!important; }}
}}
</style>
"""


if __name__ == "__main__":
    main()

"""
🌌 Vibe Oracle — Full Streamlit App
Detects emotional vibe of user input text with a mystical, visually rich UI.

Stack: streamlit==1.34.0 | nltk==3.8.1 | scikit-learn | pandas | numpy | joblib | deep-translator
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import os
import tempfile

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ── Download required NLTK data ───────────────────────────────────────────────
for _pkg in ["vader_lexicon", "stopwords", "wordnet", "punkt", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

# =============================================================================
# EMOTION DATA DICTIONARIES
# =============================================================================

EMOTION_KEYWORDS = {
    "joy": [
        "happy", "happiness", "joyful", "excited", "love", "wonderful", "amazing",
        "great", "fantastic", "cheerful", "delighted", "thrilled", "bliss", "elated",
        "ecstatic", "glad", "laugh", "smile", "celebrate", "fun", "enjoy", "grateful",
        "awesome", "brilliant", "content", "pleased", "radiant", "euphoric", "good",
        "excellent", "positive", "hopeful", "vibrant", "alive", "bright",
    ],
    "anger": [
        "angry", "anger", "furious", "rage", "mad", "hate", "irritated", "annoyed",
        "outraged", "frustrated", "enraged", "livid", "fuming", "hostile", "bitter",
        "resentful", "aggressive", "violent", "disgusted", "infuriated", "explode",
        "boiling", "seething", "wrathful", "irate", "temper", "snap", "explosive",
    ],
    "sadness": [
        "sad", "unhappy", "depressed", "miserable", "heartbroken", "grief", "cry",
        "sorrow", "mournful", "hopeless", "lonely", "gloomy", "melancholy", "despair",
        "desolate", "tragic", "painful", "lost", "tears", "devastated", "anguish",
        "down", "blue", "broken", "suffering", "hurt", "empty", "void", "miss",
    ],
    "fear": [
        "afraid", "fear", "scared", "terrified", "anxious", "nervous", "panic",
        "dread", "horror", "terror", "phobia", "worried", "uneasy", "apprehensive",
        "trembling", "fright", "nightmare", "shock", "startled", "petrified",
        "paranoid", "shaking", "tremble", "spooked", "creepy", "haunted",
    ],
    "disgust": [
        "disgusting", "gross", "revolting", "nasty", "awful", "yuck", "repulsed",
        "sick", "vomit", "nauseating", "horrible", "repulsive", "filthy", "foul",
        "unpleasant", "loathe", "abhorrent", "putrid", "hideous", "revolted",
        "sickening", "vile", "repugnant", "offensive", "stink",
    ],
    "surprise": [
        "surprised", "shocked", "astonished", "amazed", "unexpected", "wow",
        "unbelievable", "incredible", "stunning", "remarkable", "astounded",
        "speechless", "gasp", "omg", "whoa", "sudden", "startling", "jaw-dropping",
        "mind-blowing", "extraordinary", "unreal", "whoah", "no way",
    ],
}

# =============================================================================
# THREE-LANGUAGE DICTIONARY  (English · Bengali · Hindi)
# Used both for phrase-matching detection AND for the UI reference table.
# =============================================================================

# Each entry: { "en": [...], "bn": [...], "hi": [...] }
LANG_DICT = {
    "joy": {
        "en": [
            "happy", "joyful", "excited", "wonderful", "amazing", "delighted",
            "thrilled", "ecstatic", "cheerful", "blissful", "elated", "radiant",
            "grateful", "celebrate", "euphoric", "content", "pleased",
        ],
        "bn": [
            "খুশি",           # khushi  – happy
            "আনন্দিত",        # anandita – joyful
            "উত্সাহিত",       # utsahit – excited
            "দারুণ",          # darun – wonderful
            "অসাধারণ",        # asadharan – amazing
            "আনন্দ",          # ananda – joy
            "হাসি",           # hashi – smile/laughter
            "ভালো লাগছে",     # bhalo lagche – feeling good
            "অনেক মজা",       # onek moja – so much fun
            "খুব ভালো",       # khub bhalo – very good
            "উল্লাস",         # ullas – delight
            "তৃপ্তি",         # tripti – contentment
            "কৃতজ্ঞ",         # kritagya – grateful
            "উচ্ছ্বাস",       # uchchhas – elation
            "মজাদার",         # mojadaar – enjoyable
        ],
        "hi": [
            "खुश",            # khush – happy
            "खुशी",           # khushi – happiness
            "प्रसन्न",        # prasann – pleased
            "आनंदित",         # aanandित – joyful
            "उत्साहित",       # utsaahit – excited
            "शानदार",         # shaandaar – wonderful
            "मस्त",           # mast – awesome
            "जश्न",           # jashn – celebration
            "कमाल",           # kamaal – amazing
            "बेहतरीन",        # behtareen – excellent
            "दिल खुश",        # dil khush – heart happy
            "बहुत मज़ा",       # bahut maza – so much fun
            "कृतज्ञ",         # kritagna – grateful
            "उल्लास",         # ullaas – joy
            "तृप्त",          # trupt – content
        ],
    },
    "anger": {
        "en": [
            "angry", "furious", "rage", "mad", "hate", "irritated", "annoyed",
            "outraged", "frustrated", "enraged", "livid", "fuming", "hostile",
            "bitter", "resentful", "aggressive", "infuriated", "wrathful",
        ],
        "bn": [
            "রাগান্বিত",      # raganvit – angry
            "ক্রোধিত",        # krodhit – furious
            "রাগ হচ্ছে",      # rag hochhe – feeling angry
            "খুব রাগ",        # khub rag – very angry
            "বিরক্ত",         # birakt – irritated
            "ঘেন্না",         # ghenna – disgust/hatred
            "ক্ষুব্ধ",        # khubdh – outraged
            "জ্বলছি",         # jolchi – burning (with anger)
            "অসহ্য",          # asahya – unbearable
            "হতাশ",           # hatash – frustrated
            "শত্রুতা",        # shotrutha – hostility
            "প্রতিশোধ",       # protishod – revenge
            "উগ্র",           # ugro – aggressive
            "তিক্ততা",        # tiktota – bitterness
            "রোষ",            # rosh – wrath
        ],
        "hi": [
            "गुस्सा",         # gussa – angry
            "क्रोध",          # krodh – anger/rage
            "नाराज़",         # naraaz – displeased
            "चिढ़",           # chidh – irritated
            "भड़का हुआ",      # bhadka hua – enraged
            "आग बबूला",       # aag babula – furious
            "बहुत गुस्सा",    # bahut gussa – very angry
            "जलन",            # jalan – burning anger
            "कोप",            # kop – wrath
            "आक्रोश",         # aakrosh – outrage
            "नफ़रत",          # nafrat – hatred
            "झुंझलाहट",       # jhunjhlahat – annoyance
            "कड़वाहट",        # kadwahat – bitterness
            "रोष",            # rosh – fury
            "तैश",            # taish – rage
        ],
    },
    "sadness": {
        "en": [
            "sad", "unhappy", "depressed", "miserable", "heartbroken", "grief",
            "sorrow", "hopeless", "lonely", "gloomy", "melancholy", "despair",
            "devastated", "anguish", "broken", "suffering", "empty", "lost",
        ],
        "bn": [
            "দুঃখিত",          # dukhit – sad
            "মন খারাপ",        # mon kharap – feeling down
            "কষ্ট পাচ্ছি",    # koshto pachhi – feeling hurt
            "কান্না পাচ্ছে",  # kanna pachhe – feeling like crying
            "একা",             # eka – alone/lonely
            "হতাশ",            # hatash – hopeless
            "দুঃখ",            # dukh – grief
            "বিষণ্ণ",          # bishonno – gloomy
            "ভেঙে পড়েছি",    # bhenge porechhi – broken down
            "বিষাদ",           # bishad – melancholy
            "শোক",             # shok – mourning
            "যন্ত্রণা",        # jantrana – anguish
            "অসহায়",          # asahay – helpless
            "ক্লান্ত",         # klant – exhausted/weary
            "বিধ্বস্ত",        # bidhwosto – devastated
        ],
        "hi": [
            "उदास",            # udaas – sad
            "दुखी",            # dukhi – unhappy
            "निराश",           # niraash – disappointed/hopeless
            "अकेला",           # akela – lonely
            "टूटा हुआ",        # toota hua – broken
            "रोना आ रहा",      # rona aa raha – feeling like crying
            "दर्द",            # dard – pain
            "गम",              # gham – grief
            "बर्बाद",          # barbaad – devastated
            "दिल टूट गया",    # dil toot gaya – heartbroken
            "उजड़ा हुआ",      # ujda hua – desolate
            "विषाद",           # vishad – melancholy
            "पीड़ा",           # peeda – suffering
            "तकलीफ़",         # takleef – distress
            "बेबस",            # bebas – helpless
        ],
    },
    "fear": {
        "en": [
            "afraid", "scared", "terrified", "anxious", "nervous", "panic",
            "dread", "horror", "terror", "worried", "uneasy", "petrified",
            "paranoid", "trembling", "haunted", "spooked", "fright", "nightmare",
        ],
        "bn": [
            "ভয় পাচ্ছি",      # bhoy pachhi – feeling scared
            "ভয় লাগছে",       # bhoy lagche – feeling fear
            "আতঙ্কিত",         # aatankito – terrified
            "নার্ভাস",         # nervous – nervous
            "উদ্বিগ্ন",        # udbigno – anxious
            "ভয়ংকর",          # bhoyonkor – horrifying
            "দুশ্চিন্তা",      # dushchinta – worry
            "আতঙ্ক",           # aatank – panic/terror
            "শিউরে উঠছি",     # shiure uthchi – shuddering
            "ভূত ভূত",         # bhoot bhoot – ghostly
            "ত্রাস",           # tras – dread
            "শঙ্কা",           # shanka – apprehension
            "কাঁপছি",          # kampchi – trembling
            "দুঃস্বপ্ন",       # duhswapno – nightmare
            "আশঙ্কা",          # ashanka – fearful anticipation
        ],
        "hi": [
            "डर",              # dar – fear
            "डरा हुआ",         # dara hua – scared
            "घबराहट",          # ghabrahat – nervousness
            "भय",              # bhay – dread
            "आतंक",            # aatank – terror
            "चिंता",           # chinta – anxiety/worry
            "दहशत",            # dahshat – horror
            "सहम गया",         # saham gaya – startled/froze
            "कांप रहा हूं",    # kaanp raha hun – trembling
            "बहुत डर",         # bahut dar – very scared
            "डर लग रहा है",   # dar lag raha hai – feeling scared
            "भूत जैसा",        # bhoot jaisa – like a ghost
            "घबराया हुआ",      # ghabraya hua – panicked
            "रूह काँप गई",     # rooh kaanp gayi – soul trembled
            "खौफ़",            # khauf – terror
        ],
    },
    "disgust": {
        "en": [
            "disgusting", "gross", "revolting", "nasty", "yuck", "repulsed",
            "nauseating", "horrible", "repulsive", "filthy", "foul", "loathe",
            "abhorrent", "putrid", "vile", "sickening", "offensive", "stink",
        ],
        "bn": [
            "ঘেন্না লাগছে",   # ghenna lagche – feeling disgusted
            "বিরক্তিকর",       # birktikar – disgusting
            "নোংরা",           # nongra – filthy
            "বাজে",            # baje – awful/bad
            "অসহ্য গন্ধ",     # asahya gondho – unbearable smell
            "ছি ছি",           # chhi chhi – ugh/yuck
            "জঘন্য",           # jaghonyo – heinous/disgusting
            "বমি পাচ্ছে",      # bomi pachhe – feeling like vomiting
            "ঘৃণা",            # ghrina – hatred/revulsion
            "অরুচিকর",         # oruchikor – distasteful
            "ভয়াবহ",          # bhoyaboho – horrible
            "কুৎসিত",          # kutsit – ugly/repulsive
            "দুর্গন্ধ",        # durgondho – foul smell
            "বীভৎস",           # bibhotso – grotesque
            "ঘৃণ্য",           # ghrinyo – repugnant
        ],
        "hi": [
            "घिनौना",          # ghinauna – disgusting
            "बेकार",           # bekaar – useless/awful
            "गंदा",            # ganda – dirty/filthy
            "उल्टी आ रही",    # ulti aa rahi – feeling like vomiting
            "घृणा",            # ghrina – revulsion
            "बदबूदार",         # badbudaar – stinking
            "भयानक",           # bhayanak – horrible
            "नफ़रत",           # nafrat – loathing
            "छी छी",           # chhi chhi – yuck
            "गंदगी",           # gandagi – filth
            "वाहियात",         # waahiyaat – disgusting/worthless
            "बकवास",           # bakwaas – nonsense/awful
            "जुगुप्सा",        # jugupsa – disgust
            "घिन",             # ghin – revulsion
            "बेहूदा",          # behooda – absurd/repulsive
        ],
    },
    "surprise": {
        "en": [
            "surprised", "shocked", "astonished", "amazed", "unexpected", "wow",
            "unbelievable", "incredible", "stunning", "remarkable", "astounded",
            "speechless", "gasp", "mind-blowing", "extraordinary", "whoa", "omg",
        ],
        "bn": [
            "অবাক",            # obak – surprised
            "আশ্চর্য",         # ashchoryo – astonished
            "অবিশ্বাস্য",      # obishwasyo – unbelievable
            "চমকে গেছি",       # chomke gechi – startled
            "এটা কী করে সম্ভব", # eta ki kore shombhob – how is this possible
            "অদ্ভুত",          # odbhut – strange/unexpected
            "হতবাক",           # hotobak – speechless
            "বিস্মিত",         # bismit – amazed
            "চমৎকার",          # chomotkar – wonderful/astonishing
            "অপ্রত্যাশিত",     # oprottashit – unexpected
            "অকল্পনীয়",       # okalponiyo – unimaginable
            "আরে বাবা",        # are baba – oh my goodness
            "কী আশ্চর্য",      # ki ashchoryo – how surprising
            "বিষ্ময়",          # bishmoyo – amazement
            "থমকে গেছি",       # thomke gechi – stunned
        ],
        "hi": [
            "हैरान",           # hairaan – surprised
            "चौंक गया",        # chaunk gaya – startled
            "अविश्वसनीय",     # avishvasneey – unbelievable
            "अरे वाह",         # are waah – oh wow
            "क्या बात है",     # kya baat hai – what a thing
            "अचंभा",           # achambha – astonishment
            "दंग रह गया",      # dang reh gaya – stunned
            "सच में",          # sach mein – really?
            "यकीन नहीं होता", # yakeen nahi hota – can't believe it
            "ओह माय गॉड",     # oh my god
            "कमाल है",         # kamaal hai – amazing
            "अजीब",            # ajeeb – strange/unexpected
            "विस्मय",          # vismay – wonder
            "हक्का बक्का",     # hakka bakka – dumbfounded
            "अप्रत्याशित",    # apratyashit – unexpected
        ],
    },
}

# Flatten LANG_DICT into MULTILANG_PHRASES for detection engine
# (combines Bengali script + Hindi script + romanised Hinglish phrases)
_HINGLISH_EXTRA = {
    "joy":      ["bahut maza", "kitna maza", "maja aa gaya", "full masti",
                 "dil khush", "ek number", "bhai wah", "acha lag raha", "bohot khushi"],
    "anger":    ["bahut gussa", "bura lag raha", "kuch nahi chahiye", "bohot bura",
                 "chup raho", "teri toh", "faltu baat", "kya bakwas", "dimag mat kha"],
    "sadness":  ["bahut dukh", "rona aa raha", "dil toot gaya", "ek dum sad",
                 "kuch nahi ho raha", "akele hain", "bahut bura lag raha"],
    "fear":     ["bahut dar lag raha", "dar gaya", "dara hua", "bhoot jaisa",
                 "itna darna", "andhera"],
    "disgust":  ["chhi chhi", "yuck yaar", "kya bakwas hai", "bilkul pasand nahi",
                 "ganda hai", "ulti aa rahi"],
    "surprise": ["arre wah", "yaar kya baat", "sach mein", "aisa kaise",
                 "oh my god yaar", "kya hua", "kitni badi baat"],
}

MULTILANG_PHRASES = {
    emotion: (
        LANG_DICT[emotion]["bn"]
        + LANG_DICT[emotion]["hi"]
        + _HINGLISH_EXTRA[emotion]
    )
    for emotion in LANG_DICT
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

TAGLINES = {
    "joy":      "The stars smile with you tonight ✨",
    "anger":    "The cosmos feels your fire 🔥",
    "sadness":  "Even the moon weeps sometimes 🌧️",
    "fear":     "The void is vast, but you are not alone 🌒",
    "disgust":  "Some energies simply do not belong 🍄",
    "surprise": "The universe loves to astonish 💫",
}

EMOTIONS = list(EMOTION_KEYWORDS.keys())

# =============================================================================
# NLP UTILITIES
# =============================================================================

_lemmatizer = WordNetLemmatizer()
_sia        = SentimentIntensityAnalyzer()

try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = set()


def preprocess(text: str) -> str:
    """Lowercase → strip punctuation → tokenize → remove stopwords → lemmatize → rejoin."""
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words]
    return " ".join(tokens)


def translate_to_english(text: str) -> str:
    """Auto-detect source language and translate to English via deep_translator."""
    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        return result if result else text
    except Exception:
        return text


# =============================================================================
# SKLEARN ML MODEL  (TF-IDF + Logistic Regression, cached with joblib)
# =============================================================================

_MODEL_CACHE_PATH = os.path.join(tempfile.gettempdir(), "vibe_oracle_model.joblib")


def _build_training_corpus() -> pd.DataFrame:
    """
    Build a synthetic training DataFrame from keyword + phrase seeds.
    Returns a pandas DataFrame with columns ['text', 'label'].
    """
    # Sentence templates per keyword
    templates = [
        "I feel {w} today",
        "This makes me feel {w}",
        "Feeling so {w} right now",
        "I am completely {w}",
        "Everything feels {w}",
        "Such a {w} moment",
        "I cannot help but feel {w}",
        "It was truly {w}",
        "The {w} inside me is overwhelming",
        "So much {w}",
        "{w} is all I feel",
        "{w}",
    ]

    records = []
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            for tpl in templates:
                records.append({"text": tpl.format(w=kw), "label": emotion})
        # Also seed with multi-language phrases
        for phrase in MULTILANG_PHRASES.get(emotion, []):
            records.append({"text": phrase, "label": emotion})

    # Build DataFrame and shuffle with numpy for reproducibility
    df  = pd.DataFrame(records)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    df  = df.iloc[idx].reset_index(drop=True)
    return df


def _train_model():
    """Train TF-IDF + LogisticRegression pipeline; return (pipeline, label_encoder)."""
    df = _build_training_corpus()
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    X  = df["text"].apply(preprocess)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe, le


@st.cache_resource(show_spinner=False)
def get_model():
    """Return cached (pipeline, label_encoder). Train once; persist via joblib."""
    if os.path.exists(_MODEL_CACHE_PATH):
        try:
            return joblib.load(_MODEL_CACHE_PATH)
        except Exception:
            pass   # corrupt cache — retrain

    pipe, le = _train_model()
    try:
        joblib.dump((pipe, le), _MODEL_CACHE_PATH)
    except Exception:
        pass
    return pipe, le


# =============================================================================
# EMOTION DETECTION  (3-layer fusion)
# =============================================================================

def _rule_based_scores(raw_text: str, translated: str) -> dict:
    """Layer 1 + 2: multi-lang phrase hits (weight ×2) + keyword hits."""
    scores    = {e: 0.0 for e in EMOTIONS}
    raw_lower = raw_text.lower()

    # 1. Multi-language phrase detection on original text
    for emotion, phrases in MULTILANG_PHRASES.items():
        for phrase in phrases:
            if phrase.lower() in raw_lower:
                scores[emotion] += 2.0

    # 2. Keyword matching on translated + preprocessed text
    tokens        = set(preprocess(translated).split())
    translated_lc = translated.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in tokens or kw in translated_lc:
                scores[emotion] += 1.0

    return scores


def detect_emotion(raw_text: str) -> dict:
    """
    Multi-layer emotion detection → probability distribution over 6 emotions.

    Priority chain:
      1. Rule-based (phrase + keyword) — normalised if any signal
      2. ML model (TF-IDF + LogReg) probability vector
      3. VADER compound fallback when ML confidence is low
      4. Blend: 60% rule + 40% ML when rule has signal; 100% ML otherwise
    """
    # Translate once
    translated = translate_to_english(raw_text)

    # ── Layer 1 & 2: rule-based ───────────────────────────────────────────────
    rule_s     = _rule_based_scores(raw_text, translated)
    rule_total = sum(rule_s.values())

    # ── Layer 2: ML ───────────────────────────────────────────────────────────
    pipe, le      = get_model()
    processed     = preprocess(translated)
    ml_proba_arr  = pipe.predict_proba([processed])[0]          # numpy array
    ml_classes    = le.inverse_transform(np.arange(len(ml_proba_arr)))
    ml_scores     = {cls: float(ml_proba_arr[i])
                     for i, cls in enumerate(ml_classes)}
    for e in EMOTIONS:
        ml_scores.setdefault(e, 0.0)

    # ── Blend ────────────────────────────────────────────────────────────────
    if rule_total > 0:
        rule_proba = {e: rule_s[e] / rule_total for e in EMOTIONS}
        blended    = {e: 0.6 * rule_proba[e] + 0.4 * ml_scores[e] for e in EMOTIONS}
    else:
        blended = dict(ml_scores)
        # ── Layer 3: VADER safety-net when ML is uncertain ────────────────────
        top_conf = max(blended.values())
        if top_conf < 0.40:
            compound = _sia.polarity_scores(translated)["compound"]
            if compound >= 0.05:
                blended["joy"]     = blended.get("joy", 0)     + 0.50
            elif compound <= -0.05:
                blended["sadness"] = blended.get("sadness", 0) + 0.50
            total = sum(blended.values())
            blended = {e: v / total for e, v in blended.items()}

    # Final normalisation
    total = sum(blended.values())
    if total > 0:
        blended = {e: round(blended[e] / total, 4) for e in EMOTIONS}
    else:
        blended = {e: round(1.0 / len(EMOTIONS), 4) for e in EMOTIONS}

    return blended


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🌌 Vibe Oracle",
    page_icon="🔮",
    layout="centered",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Raleway:ital,wght@0,300;0,400;0,600;1,300&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #000010 !important;
    color: #f0e6ff !important;
    font-family: 'Raleway', sans-serif;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stMain"],
section.main { background: transparent !important; }

/* ── Cosmic nebula background ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse at 20% 30%, rgba(80,0,120,.35)  0%, transparent 55%),
        radial-gradient(ellipse at 80% 70%, rgba(0,50,120,.30)  0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(10,0,40,.90)   0%, transparent 100%);
    pointer-events: none; z-index: 0;
}

/* ── Star field ── */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed; inset: 0;
    background-image:
        radial-gradient(1px   1px   at 10% 15%,  white,                transparent),
        radial-gradient(1px   1px   at 25% 40%,  rgba(255,255,255,.8), transparent),
        radial-gradient(1.5px 1.5px at 40% 10%,  white,                transparent),
        radial-gradient(1px   1px   at 55% 60%,  rgba(255,255,255,.6), transparent),
        radial-gradient(1px   1px   at 70% 25%,  white,                transparent),
        radial-gradient(2px   2px   at 85% 45%,  rgba(255,255,255,.9), transparent),
        radial-gradient(1px   1px   at 15% 75%,  rgba(255,255,255,.7), transparent),
        radial-gradient(1.5px 1.5px at 60% 85%,  white,                transparent),
        radial-gradient(1px   1px   at 90% 80%,  rgba(255,255,255,.8), transparent),
        radial-gradient(1px   1px   at 35% 55%,  rgba(255,255,255,.5), transparent),
        radial-gradient(1px   1px   at 48% 33%,  rgba(200,180,255,.7), transparent),
        radial-gradient(1.5px 1.5px at 72% 58%,  rgba(180,220,255,.6), transparent);
    pointer-events: none; z-index: 0;
    animation: twinkle 4s ease-in-out infinite alternate;
}
@keyframes twinkle {
    0%   { opacity: .55; }
    100% { opacity: 1.0; }
}

/* ── Content wrapper ── */
.main-wrapper {
    position: relative; z-index: 1;
    text-align: center;
    padding: 1rem 0 2.5rem;
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
    letter-spacing: .04em;
    margin-bottom: .15rem;
    animation: titleGlow 3s ease-in-out infinite alternate;
}
@keyframes titleGlow {
    0%   { filter: drop-shadow(0 0  8px rgba(192,132,252,.6));  }
    100% { filter: drop-shadow(0 0 22px rgba(56,189,248,.85));  }
}

/* ── Subtitle ── */
.oracle-subtitle {
    font-size: 1.15rem; color: #c4b5fd;
    letter-spacing: .18em; margin-bottom: 1.6rem; font-weight: 300;
}

/* ── Floating fairy ── */
.fairy-float {
    font-size: 2.8rem; display: block;
    animation: fairyFloat 2.5s ease-in-out infinite;
    margin-bottom: .4rem;
}
@keyframes fairyFloat {
    0%,100% { transform: translateY(0)     rotate(-5deg); }
    50%      { transform: translateY(-14px) rotate(5deg);  }
}

/* ── Spinning moon ── */
.moon-spin {
    display: inline-block; font-size: 2rem;
    animation: moonSpin 9s linear infinite;
}
@keyframes moonSpin {
    from { transform: rotate(0deg);   }
    to   { transform: rotate(360deg); }
}

/* ── Aura card ── */
.aura-card {
    border-radius: 20px; padding: 2rem 1.5rem;
    margin: 1.6rem auto; max-width: 600px;
    backdrop-filter: blur(14px);
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.10);
    transition: box-shadow .5s ease;
}

/* ── Emotion name ── */
.emotion-name {
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(1.8rem, 5vw, 3rem);
    font-weight: 700; letter-spacing: .08em; margin: .4rem 0;
    animation: emotionPulse 2s ease-in-out infinite alternate;
}
@keyframes emotionPulse {
    0%   { filter: brightness(1);                                    }
    100% { filter: brightness(1.35) drop-shadow(0 0 14px currentColor); }
}

.emotion-emoji-big {
    font-size: 3.5rem; display: block; margin: .3rem 0;
    animation: emojiBounce 1.2s ease-in-out infinite;
}
@keyframes emojiBounce {
    0%,100% { transform: scale(1);   }
    50%      { transform: scale(1.2); }
}

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    font-size: .78rem; font-weight: 600; letter-spacing: .12em;
    padding: .25rem .75rem; border-radius: 999px;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.15);
    color: #e2d9f3; margin-top: .5rem;
    font-family: 'Raleway', sans-serif;
}

/* ── Emotion breakdown bars ── */
.bar-row {
    display: flex; align-items: center;
    gap: 10px; margin: 7px 0;
    font-family: 'Raleway', sans-serif;
}
.bar-label {
    width: 95px; text-align: right;
    font-size: .82rem; color: #e2d9f3;
    font-weight: 600; text-transform: capitalize;
}
.bar-track {
    flex: 1; height: 16px;
    background: rgba(255,255,255,.07);
    border-radius: 8px; overflow: hidden;
}
.bar-fill {
    height: 100%; border-radius: 8px;
    animation: barGrow 1.3s cubic-bezier(.23,1,.32,1) forwards;
}
@keyframes barGrow {
    from { width: 0%;             }
    to   { width: var(--bar-w); }
}
.bar-pct {
    width: 44px; font-size: .8rem;
    color: #c4b5fd; font-weight: 600;
}

/* ── Section label ── */
.section-label {
    font-family: 'Cinzel Decorative', serif;
    font-size: .82rem; letter-spacing: .2em;
    color: #a78bfa; text-transform: uppercase; margin-bottom: .8rem;
}

/* ── Divider ── */
.mystic-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,.5), transparent);
    margin: 1.5rem 0;
}

/* ── Warning ── */
.warn-msg {
    color: #fbbf24; font-size: 1rem;
    padding: .8rem 1.2rem;
    border: 1px solid rgba(251,191,36,.3); border-radius: 10px;
    background: rgba(251,191,36,.08); margin-top: 1rem;
}

/* ── Streamlit widget overrides ── */
textarea {
    background: rgba(255,255,255,.05) !important;
    color: #f0e6ff !important;
    border: 1px solid rgba(168,85,247,.4) !important;
    border-radius: 12px !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 1rem !important;
}
textarea:focus {
    border-color: rgba(168,85,247,.9) !important;
    box-shadow: 0 0 20px rgba(168,85,247,.3) !important;
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #0ea5e9) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; padding: .7rem 2.5rem !important;
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 1.05rem !important; letter-spacing: .05em !important;
    cursor: pointer !important; transition: all .3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,.5) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,.7) !important;
}

/* ── Expander dark theme ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(168,85,247,.3) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #c4b5fd !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: .95rem !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LAYOUT
# =============================================================================

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="oracle-title">🌌 Vibe Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="oracle-subtitle">Speak your vibe 🌙</p>', unsafe_allow_html=True)

# ── Warm the model once (cached after first run) ──────────────────────────────
with st.spinner("🔭 Aligning the cosmic model…"):
    _pipe, _le = get_model()

# ── Text input ────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="",
    placeholder="Type anything… in any language 🔮",
    height=130,
    key="vibe_input",
    label_visibility="collapsed",
)

c1, c2, c3 = st.columns([2, 1.5, 2])
with c2:
    reveal = st.button("🔮 Reveal the Vibe", use_container_width=True)

# ── Detection & output ────────────────────────────────────────────────────────
if reveal:
    if not user_input.strip():
        st.markdown(
            '<div class="warn-msg">'
            '⚠️ Please enter some text to reveal your vibe…'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("✨ Reading the cosmic vibrations…"):
            scores = detect_emotion(user_input)

        # Use pandas Series for ordering / max lookup
        score_series = pd.Series(scores, dtype=float).sort_values(ascending=False)
        dominant     = str(score_series.idxmax())
        dominant_pct = int(round(score_series[dominant] * 100))

        color = EMOTION_COLORS[dominant]
        emoji = EMOTION_EMOJIS[dominant]
        fairy = FAIRY_EMOJIS[dominant]
        moon  = MOON_PHASES[dominant]
        tag   = TAGLINES[dominant]

        # ── Aura card ─────────────────────────────────────────────────────────
        glow = (
            f"0 0 60px {color}55, "
            f"0 0 120px {color}22, "
            f"inset 0 0 40px {color}11"
        )
        st.markdown(f"""
        <div class="aura-card" style="box-shadow:{glow}; border-color:{color}44;">
            <span class="fairy-float">{fairy}</span>
            <span class="moon-spin">{moon}</span>
            <p class="section-label">Your Dominant Vibe</p>
            <span class="emotion-emoji-big">{emoji}</span>
            <p class="emotion-name" style="color:{color};">{dominant.upper()}</p>
            <p style="color:#c4b5fd; font-size:.95rem; margin-top:.2rem;
                      font-family:'Raleway',sans-serif;">
                {fairy} &nbsp; {moon} &nbsp; {emoji}
            </p>
            <span class="conf-badge">✦ Confidence: {dominant_pct}% ✦</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Emotion breakdown bars ────────────────────────────────────────────
        st.markdown('<hr class="mystic-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label" style="text-align:center;">'
            '✦ Emotion Breakdown ✦</p>',
            unsafe_allow_html=True,
        )

        # Iterate emotions in descending order (pandas sort already done)
        for emotion in score_series.index.tolist():
            pct = int(round(score_series[emotion] * 100))
            ec  = EMOTION_COLORS[emotion]
            ee  = EMOTION_EMOJIS[emotion]

            if pct == 100:
                fill     = f"linear-gradient(90deg, {ec}, #fff8)"
                glow_bar = f"0 0 10px {ec}"
            else:
                fill     = f"linear-gradient(90deg, {ec}cc, {ec}44)"
                glow_bar = f"0 0 6px {ec}88"

            st.markdown(f"""
            <div class="bar-row">
                <span class="bar-label">{ee} {emotion}</span>
                <div class="bar-track">
                    <div class="bar-fill"
                         style="--bar-w:{pct}%;
                                width:{pct}%;
                                background:{fill};
                                box-shadow:{glow_bar};">
                    </div>
                </div>
                <span class="bar-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Mystical tagline ──────────────────────────────────────────────────
        st.markdown(f"""
        <p style="text-align:center; margin-top:1.2rem;
                  font-style:italic; color:#a78bfa;
                  font-size:.95rem; letter-spacing:.06em;
                  font-family:'Raleway',sans-serif;">
            {tag}
        </p>
        """, unsafe_allow_html=True)

        # ── Three-language dictionary panel ───────────────────────────────────
        st.markdown('<hr class="mystic-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label" style="text-align:center;">'
            '✦ Emotion Dictionary ✦ English · বাংলা · हिंदी</p>',
            unsafe_allow_html=True,
        )

        d_color = EMOTION_COLORS[dominant]
        d_entry = LANG_DICT[dominant]

        # Build a 3-column word-chip layout
        def _chips(words: list, chip_color: str) -> str:
            chips = "".join(
                f'<span style="display:inline-block; margin:3px 4px; padding:4px 10px;'
                f'border-radius:999px; font-size:.78rem; font-weight:600;'
                f'background:{chip_color}22; border:1px solid {chip_color}55;'
                f'color:#f0e6ff; font-family:\'Raleway\',sans-serif;">'
                f'{w}</span>'
                for w in words
            )
            return chips

        lang_labels = {"en": "🇬🇧 English", "bn": "🇧🇩 বাংলা", "hi": "🇮🇳 हिंदी"}

        col_en, col_bn, col_hi = st.columns(3)
        for col, lang_key in zip([col_en, col_bn, col_hi], ["en", "bn", "hi"]):
            with col:
                words  = d_entry[lang_key]
                chips  = _chips(words, d_color)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);
                            border:1px solid {d_color}33;
                            border-radius:14px; padding:12px 10px;
                            min-height:200px;">
                    <p style="font-family:'Cinzel Decorative',serif;
                               font-size:.72rem; letter-spacing:.15em;
                               color:{d_color}; text-align:center;
                               margin-bottom:8px; text-transform:uppercase;">
                        {lang_labels[lang_key]}
                    </p>
                    <div style="text-align:center; line-height:2;">
                        {chips}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Also show a compact table of all emotions × all 3 languages
        st.markdown('<hr class="mystic-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label" style="text-align:center;">'
            '✦ Full Emotion Lexicon ✦</p>',
            unsafe_allow_html=True,
        )

        # Build pandas DataFrame for the full table
        table_rows = []
        for emo in EMOTIONS:
            ec  = EMOTION_COLORS[emo]
            ee  = EMOTION_EMOJIS[emo]
            row = {
                "Emotion":  f"{ee} {emo.capitalize()}",
                "English":  " · ".join(LANG_DICT[emo]["en"][:5]),
                "বাংলা":    " · ".join(LANG_DICT[emo]["bn"][:5]),
                "हिंदी":    " · ".join(LANG_DICT[emo]["hi"][:5]),
            }
            table_rows.append(row)

        df_table = pd.DataFrame(table_rows)

        # Render as a styled HTML table (no st.dataframe to preserve dark theme)
        header_cells = "".join(
            f'<th style="padding:8px 12px; text-align:left; '
            f'color:#c4b5fd; font-family:\'Cinzel Decorative\',serif; '
            f'font-size:.72rem; letter-spacing:.12em; border-bottom:1px solid rgba(168,85,247,.3);">'
            f'{col}</th>'
            for col in df_table.columns
        )

        body_rows_html = ""
        for i, row in df_table.iterrows():
            emo_key = EMOTIONS[i]
            rc      = EMOTION_COLORS[emo_key]
            cells   = "".join(
                f'<td style="padding:7px 12px; font-size:.8rem; '
                f'color:#e2d9f3; font-family:\'Raleway\',sans-serif; '
                f'border-bottom:1px solid rgba(255,255,255,.05);">{v}</td>'
                for v in row.values
            )
            body_rows_html += (
                f'<tr style="background:{rc}0d;">{cells}</tr>'
            )

        st.markdown(f"""
        <div style="overflow-x:auto; border-radius:12px;
                    border:1px solid rgba(168,85,247,.25);
                    background:rgba(255,255,255,.03);">
            <table style="width:100%; border-collapse:collapse;">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{body_rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ── Always-visible expander: full dictionary reference ────────────────────────
with st.expander("📖 Browse the Full 3-Language Emotion Dictionary"):
    for emo in EMOTIONS:
        ec = EMOTION_COLORS[emo]
        ee = EMOTION_EMOJIS[emo]
        st.markdown(
            f'<p style="font-family:\'Cinzel Decorative\',serif; '
            f'font-size:.9rem; color:{ec}; margin:1rem 0 .3rem;">'
            f'{ee} {emo.upper()}</p>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        for col, lang_key, flag in [
            (c1, "en", "🇬🇧 English"),
            (c2, "bn", "🇧🇩 বাংলা"),
            (c3, "hi", "🇮🇳 हिंदी"),
        ]:
            with col:
                words_md = "\n".join(f"- {w}" for w in LANG_DICT[emo][lang_key])
                st.markdown(
                    f'<p style="font-size:.75rem; color:#a78bfa; '
                    f'font-weight:700; margin-bottom:4px;">{flag}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(words_md)

st.markdown('</div>', unsafe_allow_html=True)   # close .main-wrapper

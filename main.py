
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os, json, re, random, pickle, traceback

import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="AI/ML Chatbot (FastAPI)")

# ----------------- Static & Templates -----------------
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ----------------- Load intents -----------------
intents_path = os.path.join(BASE_DIR, "intents.json")
if not os.path.exists(intents_path):
    # minimal fallback to avoid crash
    sample = {
        "intents":[
            {"tag":"greeting","patterns":["hi","hello","hey"],"responses":["Hello!","Hi there!"]},
            {"tag":"unknown","patterns":[""],"responses":["I don't understand yet."]}
        ]
    }
    with open(intents_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)

with open(intents_path, "r", encoding="utf-8") as f:
    INTENTS = json.load(f)

# ----------------- NLTK resources -----------------
for res, path in [("punkt","tokenizers/punkt"), ("punkt_tab","tokenizers/punkt_tab"), ("stopwords","corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"NLTK resource '{res}' not found. Downloading...")
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Failed to download NLTK resource: {res} ({e})")

# ----------------- Optional spaCy for entities -----------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
        print("spaCy model 'en_core_web_sm' not found. Regex-only entity extraction will be used.")
except Exception:
    nlp = None
    print("spaCy not installed. Regex-only entity extraction will be used.")

# ----------------- NLP helpers -----------------
stemmer = SnowballStemmer("english")

def preprocess_text(text: str) -> str:
    try:
        toks = nltk.word_tokenize(text)
    except LookupError:
        toks = text.split()
    return " ".join(stemmer.stem(w.lower()) for w in toks) if text else ""

def create_model_data(intents):
    patterns, tags = [], []
    for intent in intents.get("intents", []):
        for patt in intent.get("patterns", []):
            clean = re.sub(r"%.+?%", "", patt).strip()
            if not clean:
                continue
            patterns.append(preprocess_text(clean))
            tags.append(intent.get("tag"))
    return patterns, tags

def train_intent_model(patterns, tags):
    # Stronger vectorizer for short patterns
    pipeline = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])
    pipeline.fit(patterns, tags)
    return pipeline

def top_pred(model, processed_text: str):
    # Return (label, prob) using decision_function/proba when available
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([processed_text])[0]
            classes = model.classes_
            idx = int(np.argmax(probs))
            return classes[idx], float(probs[idx])
        elif hasattr(model, "decision_function"):
            scores = model.decision_function([processed_text])[0]
            classes = model.classes_
            idx = int(np.argmax(scores))
            # map scores to pseudo-prob via softmax for readability
            exp = np.exp(scores - np.max(scores))
            probs = exp / np.sum(exp)
            return classes[idx], float(probs[idx])
        else:
            pred = model.predict([processed_text])[0]
            return pred, 1.0
    except Exception:
        return "unknown", 0.0

def get_response(tag: str) -> str:
    for intent in INTENTS.get("intents", []):
        if intent.get("tag") == tag:
            return random.choice(intent.get("responses", ["Sorry, I don't understand."]))
    return "I'm sorry, I don't understand. Can you rephrase?"

def extract_entities(text: str) -> dict:
    entities = {}
    # SpaCy entities (optional)
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("PRODUCT","ORG","GPE"):
                    entities.setdefault("product", ent.text)
        except Exception:
            pass
    # Regex entities
    m = re.search(r"\b(?:order number|order|track|package|ticket)\s*[:#]?\s*([A-Z0-9\-]{5,20})\b", text, re.I)
    if m:
        entities["order_number"] = m.group(1)
    if "product" not in entities:
        m2 = re.search(r"\b(iPhone|Galaxy|MacBook|laptop|smartphone|camera|headphones|printer|router)\b", text, re.I)
        if m2:
            entities["product"] = m2.group(1)
    return entities

# --------------- Rule-based fallback (only used when needed) ---------------
RULES = [
    (r"\b(ai|artificial intelligence)\b", "ai", "AI is the simulation of human intelligence in machines."),
    (r"\b(ml|machine learning)\b", "ml", "Machine Learning teaches computers to learn patterns from data."),
    (r"\b(nlp|natural language)\b", "nlp", "NLP helps computers understand and generate human language."),
    (r"\b(vision|computer vision)\b", "computer_vision", "Computer Vision enables machines to understand images and video."),
    (r"\b(deploy|docker|kubernetes|uvicorn)\b", "devops", "You can deploy models via REST APIs, Docker, and Kubernetes."),
    (r"\b(python|numpy|pandas)\b", "python", "Python is great for AI: start with NumPy, pandas, scikit-learn, PyTorch or TensorFlow.")
]

def rule_based_response(text: str):
    txt = text.lower()
    for pattern, tag, reply in RULES:
        if re.search(pattern, txt):
            return tag, reply
    return None, None

# ----------------- Model training / loading -----------------
patterns, tags = create_model_data(INTENTS)
unique_tags = sorted(set(tags))
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "intent_model.pkl")

USE_RULE_FALLBACK = False
intent_model = None

if len(unique_tags) >= 2 and len(patterns) == len(tags) and len(patterns) > 0:
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as mf:
                intent_model = pickle.load(mf)
            print(f"Loaded intent model from {model_path}")
        else:
            print("No saved model found — training intent model (this may take a moment)...")
            intent_model = train_intent_model(patterns, tags)
            with open(model_path, "wb") as mf:
                pickle.dump(intent_model, mf)
            print(f"Model trained and saved to {model_path}")
    except Exception as e:
        print(f"Model load/train error: {e}. Enabling rule-based fallback.")
        intent_model = None
        USE_RULE_FALLBACK = True
else:
    print(f"Warning: insufficient label variety (found {len(unique_tags)} unique tag(s)). Enabling rule-based fallback.")
    USE_RULE_FALLBACK = True

# ----------------- CORS (dev) -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Routes -----------------
# ----------------- Routes -----------------
@app.get("/")
async def root():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
@app.post("/chat")
async def chat_endpoint(payload: dict):
    try:
        text = (payload or {}).get("message", "")
        if not text or not text.strip():
            return JSONResponse({"response": "Please send a message.", "intent": "none", "confidence": 0.0, "entities": {}})

        processed = preprocess_text(text)

        # Try ML model first (if available)
        tag, conf = ("unknown", 0.0)
        if intent_model is not None:
            tag, conf = top_pred(intent_model, processed)

        # If confidence too low, try rules
        CONF_THRESHOLD = 0.35
        if intent_model is None or conf < CONF_THRESHOLD:
            rtag, rreply = rule_based_response(text)
            if rreply:
                entities = extract_entities(text)
                return JSONResponse({"response": rreply, "intent": rtag or "rule", "confidence": round(conf, 3), "entities": entities})

        # Use ML response
        response = get_response(tag)
        entities = extract_entities(text)
        for k, v in entities.items():
            response = response.replace(f"%{k}%", v)

        return JSONResponse({"response": response, "intent": tag, "confidence": round(conf, 3), "entities": entities})

    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        return JSONResponse({"response": "Sorry, something went wrong on the server.", "intent": "error", "confidence": 0.0, "entities": {}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

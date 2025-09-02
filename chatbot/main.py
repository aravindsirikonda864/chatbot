from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, json, re, random, pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="AI/ML Chatbot (FastAPI)")

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load intents
with open(os.path.join(BASE_DIR, "intents.json"), "r", encoding="utf-8") as f:
    INTENTS = json.load(f)

# Ensure NLTK resources are present (download if missing)
for res, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"NLTK resource '{res}' not found. Downloading...")
        nltk.download(res, quiet=True)

# Optional: try loading spaCy NER model for better entity extraction (graceful fallback)
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # If model is not downloaded, don't crash; user can install with `python -m spacy download en_core_web_sm`
        nlp = None
        print("spaCy model 'en_core_web_sm' not found. Entity extraction will use regex fallback.")
except Exception:
    nlp = None
    print("spaCy not installed. To improve entity extraction, install spaCy and download 'en_core_web_sm'.")

# Prepare NLP tools
stemmer = SnowballStemmer("english")

def preprocess_text(text: str) -> str:
    toks = nltk.word_tokenize(text)
    return " ".join(stemmer.stem(w.lower()) for w in toks)

def create_model_data(intents):
    patterns, tags = [], []
    for intent in intents["intents"]:
        for patt in intent.get("patterns", []):
            clean = re.sub(r"%.+?%", "", patt).strip()
            if not clean:
                continue
            patterns.append(preprocess_text(clean))
            tags.append(intent["tag"])
    return patterns, tags

def train_intent_model(patterns, tags):
    pipeline = Pipeline([
        ("vec", TfidfVectorizer()),
        ("clf", LogisticRegression(solver="liblinear", multi_class="ovr", max_iter=200))
    ])
    pipeline.fit(patterns, tags)
    return pipeline

# --- Model persistence: save/load trained pipeline to speed up startup ---
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "intent_model.pkl")

patterns, tags = create_model_data(INTENTS)

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as mf:
            intent_model = pickle.load(mf)
        print(f"Loaded intent model from {model_path}")
    except Exception as e:
        print(f"Failed to load saved model ({e}). Retraining...")
        intent_model = train_intent_model(patterns, tags)
        with open(model_path, "wb") as mf:
            pickle.dump(intent_model, mf)
        print(f"Saved newly trained model to {model_path}")
else:
    print("No saved model found — training intent model (this may take a moment)...")
    intent_model = train_intent_model(patterns, tags)
    with open(model_path, "wb") as mf:
        pickle.dump(intent_model, mf)
    print(f"Model trained and saved to {model_path}")

def get_response(tag: str) -> str:
    for intent in INTENTS["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent.get("responses", ["Sorry, I don't understand."]))
    return "I'm sorry, I don't understand. Can you rephrase?"

def extract_entities(text: str) -> dict:
    """
    Try spaCy NER first (if available), otherwise use regex fallback for common placeholders
    like order numbers. Returns a dict like {'product': 'iPhone', 'order_number': 'ABC12345'}.
    """
    entities = {}
    if nlp:
        try:
            doc = nlp(text)
            # simple mapping: PRODUCT/ORG/GPE -> product (you can expand this mapping)
            for ent in doc.ents:
                if ent.label_ in ("PRODUCT", "ORG", "GPE"):
                    # only set if not already set (keeps first found)
                    entities.setdefault("product", ent.text)
        except Exception:
            # if spacy NER errors for some reason, ignore and fallback to regex
            pass

    # Regex fallback for order numbers and simple product patterns
    m = re.search(r"\b(?:order number|order|track|package|ticket)\s*[:#]?\s*([A-Z0-9\-]{5,20})\b", text, re.I)
    if m:
        entities["order_number"] = m.group(1)

    # Try a lightweight product regex if spaCy not available or missed it
    if "product" not in entities:
        m2 = re.search(r"\b(iPhone|Galaxy|MacBook|laptop|smartphone|camera|headphones|printer|router)\b", text, re.I)
        if m2:
            entities["product"] = m2.group(1)

    return entities

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: dict):
    text = payload.get("message", "")
    processed = preprocess_text(text)
    try:
        tag = intent_model.predict([processed])[0]
    except Exception:
        tag = "unknown"
    response = get_response(tag)

    entities = extract_entities(text)
    # Replace placeholders in the response (if present)
    for k, v in entities.items():
        response = response.replace(f"%{k}%", v)

    return JSONResponse({"response": response, "intent": tag, "entities": entities})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# 🤖 AI-Powered Chatbot using FastAPI

This is an **AI-powered chatbot** built with **FastAPI** that can handle both **daily conversations** and **computer science/AI-related queries**.  
The chatbot uses **Natural Language Processing (NLP)** and a **machine learning model** (TF-IDF + Logistic Regression) trained on a custom dataset.

---

## 🚀 Features
- 🗣️ **Daily Conversations** (hello, how are you, goodbye, etc.)
- 💻 **Computer Science & AI Topics** (NLP, Machine Learning, Cloud, Databases, etc.)
- 🔎 **Intent Classification** using ML
- 🛡️ **Fallback System** for unknown queries
- 📑 **Entity Recognition** (simple order/product extraction)
- 🌐 **Web Interface** via FastAPI

---

## 🛠️ Technologies Used
- **Python 3.x**
- **FastAPI + Uvicorn**
- **Scikit-learn** (TF-IDF + Logistic Regression)
- **NLTK**
- **HTML, CSS, JavaScript**
- **JSON** for dataset

---

## 📂 Project Structure
```
chatbot1_updated/
│
├── main.py              # FastAPI app
├── intents.json         # Training dataset
├── models/              # Saved ML model & vectorizer
├── static/              # CSS, JS
├── templates/           # HTML frontend
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## ⚡ Installation & Running

1. Clone the repo:
   ```bash
   git clone https://github.com/aravindsirikonda864/chatbot.git
   cd chatbot-fastapi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows PowerShell)
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

5. Open your browser:
   ```
   http://127.0.0.1:8000
   ```

---

## 🎯 Future Improvements
- Add **Deep Learning model (RNN/Transformer)** for better accuracy.
- Enhance **entity recognition** using SpaCy.
- Deploy on **AWS / Azure / Heroku**.
- Add **voice input/output** for real-time interaction.



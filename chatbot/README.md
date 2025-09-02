
# AI/ML Chatbot - FastAPI Web App (Expanded)

This is the FastAPI-based chatbot project updated with expanded intents covering modern AI topics including:

- Natural Language Processing: Transformers, BERT, GPT, Prompt Engineering, RAG, Embeddings, Vector DBs
- Computer Vision: YOLO, Vision Transformers (ViT), Diffusion Models, Image generation concepts
- Modern ML topics: Self-supervised learning, Contrastive Learning, Transfer Learning, Federated Learning, Reinforcement Learning
- MLOps/Deployment: Docker, Edge AI, Model monitoring, Model explainability and evaluation metrics

Files to run:
- main.py
- intents.json (updated with many new topics)
- templates/index.html
- static/style.css
- requirements.txt

Run:
1. Create & activate venv
2. pip install -r requirements.txt
3. python -m nltk.downloader punkt stopwords
4. uvicorn main:app --reload --host 127.0.0.1 --port 8000

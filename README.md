# Self-Service KB Assistant (Free, No API Key)

A lightweight RAG chatbot that answers **only from your Markdown Knowledge Base**, cites sources, asks clarifying questions when uncertain, and offers quick guided intents. Built with **Gradio**, **FAISS**, **sentence-transformers**, and an **extractive QA model**—no paid API needed.

## Features
- Markdown-based KB (`/kb/*.md`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector search: FAISS (cosine on normalized vectors)
- Reader: `deepset/roberta-base-squad2` (extractive QA)
- Citations (title + section)
- Low-confidence fallback (suggest related articles)
- Quick intents (buttons for top tasks)
- One-click “Rebuild Index” admin control

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss

import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

KB_DIR = Path("./kb")
INDEX_DIR = Path("./.index")
INDEX_DIR.mkdir(exist_ok=True, parents=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
READER_MODEL_NAME = "deepset/roberta-base-squad2"

EMBEDDINGS_PATH = INDEX_DIR / "kb_embeddings.npy"
METADATA_PATH = INDEX_DIR / "kb_metadata.json"
FAISS_PATH = INDEX_DIR / "kb_faiss.index"


# ---------------------------
# Utilities: Markdown loading
# ---------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

def read_markdown_files(kb_dir: Path) -> List[Dict]:
    docs = []
    for md_path in sorted(kb_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        title = md_path.stem.replace("_", " ").title()
        # Try first H1 as title if present
        m = re.search(r"^#\s+(.*)$", text, flags=re.MULTILINE)
        if m:
            title = m.group(1).strip()

        docs.append({
            "filepath": str(md_path),
            "filename": md_path.name,
            "title": title,
            "text": text
        })
    return docs


def chunk_markdown(doc: Dict, chunk_chars: int = 1200, overlap: int = 150) -> List[Dict]:
    """
    Simple header-aware chunking: split by H2/H3 when possible and then by char length.
    Stores anchor-ish metadata for basic citations.
    """
    text = doc["text"]
    # Split by H2/H3 as sections (fallback to entire text)
    sections = re.split(r"(?=^##\s+|\n##\s+|\n###\s+|^###\s+)", text, flags=re.MULTILINE)
    if len(sections) == 1:
        sections = [text]

    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Derive a section heading for citation
        heading_match = HEADING_RE.search(sec)
        section_heading = heading_match.group(2).strip() if heading_match else doc["title"]

        # Hard wrap into chunks
        start = 0
        while start < len(sec):
            end = min(start + chunk_chars, len(sec))
            chunk_text = sec[start:end].strip()
            if chunk_text:
                chunks.append({
                    "doc_title": doc["title"],
                    "filename": doc["filename"],
                    "filepath": doc["filepath"],
                    "section": section_heading,
                    "content": chunk_text
                })
            start = end - overlap if end - overlap > 0 else end
            if start < 0:
                start = 0
            if end == len(sec):
                break

    return chunks


# ---------------------------
# Build / Load Index
# ---------------------------

class KBIndex:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL_NAME)
        self.reader = pipeline("question-answering", model=self.reader_model, tokenizer=self.reader_tokenizer)

        self.index = None  # FAISS index
        self.embeddings = None  # numpy array
        self.metadata = []  # list of dicts

    def build(self, kb_dir: Path):
        docs = read_markdown_files(kb_dir)
        if not docs:
            raise RuntimeError(f"No markdown files found in {kb_dir.resolve()}. Please add *.md files.")

        # Produce chunks
        all_chunks = []
        for d in docs:
            all_chunks.extend(chunk_markdown(d))

        texts = [c["content"] for c in all_chunks]
        if not texts:
            raise RuntimeError("No content chunks generated from KB.")

        embeddings = self.embedder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index (cosine via inner product on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.embeddings = embeddings
        self.metadata = all_chunks

        # Persist to disk
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        faiss.write_index(index, str(FAISS_PATH))

    def load(self):
        if not (EMBEDDINGS_PATH.exists() and METADATA_PATH.exists() and FAISS_PATH.exists()):
            return False

        self.embeddings = np.load(EMBEDDINGS_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.index = faiss.read_index(str(FAISS_PATH))
        return True

    def rebuild_if_kb_changed(self):
        """
        Very light heuristic: if index older than newest kb file, rebuild.
        """
        kb_mtime = max([p.stat().st_mtime for p in KB_DIR.glob("*.md")] or [0])
        idx_mtime = min([
            EMBEDDINGS_PATH.stat().st_mtime if EMBEDDINGS_PATH.exists() else 0,
            METADATA_PATH.stat().st_mtime if METADATA_PATH.exists() else 0,
            FAISS_PATH.stat().st_mtime if FAISS_PATH.exists() else 0,
        ])
        if kb_mtime > idx_mtime:
            self.build(KB_DIR)

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        indices = I[0].tolist()
        sims = D[0].tolist()
        return list(zip(indices, sims))

    def answer(self, question: str, retrieved: List[Tuple[int, float]]):
        """
        Use extractive QA across the top retrieved chunks; pick the best span by score.
        Return (answer_text, best_score, citations)
        """
        best = {"text": None, "score": -1e9, "meta": None, "ctx": None, "sim": 0.0}
        for idx, sim in retrieved:
            meta = self.metadata[idx]
            context = meta["content"]
            try:
                out = self.reader(question=question, context=context)
            except Exception:
                continue
            score = float(out.get("score", 0.0))
            if score > best["score"]:
                best = {
                    "text": out.get("answer", "").strip(),
                    "score": score,
                    "meta": meta,
                    "ctx": context,
                    "sim": float(sim)
                }

        if not best["text"]:
            return None, 0.0, []

        # Build citations: top 2 sources from retrieved
        citations = []
        seen = set()
        for idx, sim in retrieved[:2]:
            meta = self.metadata[idx]
            key = (meta["filename"], meta["section"])
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "title": meta["doc_title"],
                "filename": meta["filename"],
                "section": meta["section"]
            })
        return best["text"], best["score"], citations


kb = KBIndex()

def ensure_index():
    if not kb.load():
        kb.build(KB_DIR)
    else:
        kb.rebuild_if_kb_changed()

ensure_index()


# ---------------------------
# Clarify / Guardrails logic
# ---------------------------

def format_citations(citations: List[Dict]) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations:
        lines.append(f"• **{c['title']}** — _{c['section']}_  (`{c['filename']}`)")
    return "\n".join(lines)

LOW_CONF_THRESHOLD = 0.20     # reader score heuristic (0–1)
LOW_SIM_THRESHOLD  = 0.30     # retriever sim heuristic (cosine/IP on normalized vectors)

HELPFUL_SUGGESTIONS = [
    ("Connect WhatsApp", "How do I connect my WhatsApp number?"),
    ("Reset Password", "I can't sign in / forgot my password"),
    ("First Automation", "How do I create my first automation?"),
    ("Billing & Invoices", "How do I download invoices for billing?"),
    ("Fix Instagram Connect", "Why can't I connect Instagram?")
]


def respond(user_msg, history):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return "How can I help? Try: **Connect WhatsApp** or **Reset password**."

    # Retrieve
    retrieved = kb.retrieve(user_msg, top_k=4)
    if not retrieved:
        return "I couldn't find anything yet. Try rephrasing or pick a quick action below."

    # Answer
    span, score, citations = kb.answer(user_msg, retrieved)

    # If no span, surface top articles as fallback
    if not span:
        suggestions = "\n".join([f"- {c['title']} — _{c['section']}_" for c in citations]) or "- Try a different query."
        return f"I’m not fully sure. Here are the closest matches:\n\n{suggestions}"

    # Confidence heuristics
    best_sim = max(_

"""
LawyerBot — api.py
============================
College    : Indira Gandhi Institute of Engineering and Technology
University : APJ Abdul Kalam Technological University
Developers : Muhammed Sadik, Arshad Basheer, Alimon KR

STACK (matches your existing project):
  - FAISS (vectordb/lawyerbot.index)   ← your existing index, no rebuild needed
  - metadata.json                       ← your existing metadata
  - all-MiniLM-L6-v2                   ← same model you used in vector.py
  - Groq API (llama-3.3-70b-versatile) ← same model from rag_groq.py
  - FastAPI                             ← backend server

HOW TO RUN:
  1. Create a .env file in PythonProject/ with:
         GROQ_API_KEY=your_groq_key_here

  2. Run from PythonProject/ folder:
         uvicorn api:app --reload

  3. Open lawyerbot.html in your browser

NOTE: Your vectordb, chunks, cleaned_text folders are used as-is.
      No need to reprocess or rebuild anything.
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq

# ══════════════════════════════════════════
#  PATHS — points to YOUR existing folders
# ══════════════════════════════════════════
BASE_DIR     = Path(r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset")
VECTOR_DB    = BASE_DIR / "vectordb" / "lawyerbot.index"
METADATA     = BASE_DIR / "vectordb" / "metadata.json"
CHUNKS_DIR   = BASE_DIR / "chunks"

# How many top chunks to retrieve per question
TOP_K = 5

# ══════════════════════════════════════════
#  LOAD ENV & CLIENTS
# ══════════════════════════════════════════
load_dotenv(dotenv_path=r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\PythonProject\.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found!\n"
        "Create a .env file in PythonProject/ with:\n"
        "GROQ_API_KEY=your_key_here\n"
        "Get a free key at: https://console.groq.com"
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# ══════════════════════════════════════════
#  LOAD MODELS & VECTOR DB AT STARTUP
# ══════════════════════════════════════════
print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model loaded")

print("Loading FAISS index...")
if not VECTOR_DB.exists():
    raise RuntimeError(
        f"FAISS index not found at: {VECTOR_DB}\n"
        "Run vector.py first to build the index."
    )
faiss_index = faiss.read_index(str(VECTOR_DB))
print(f"✓ FAISS index loaded — {faiss_index.ntotal} vectors")

print("Loading metadata...")
if not METADATA.exists():
    raise RuntimeError(f"metadata.json not found at: {METADATA}")
with open(METADATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"✓ Metadata loaded — {len(metadata)} chunks")

# ══════════════════════════════════════════
#  CATEGORY → SOURCE FILE MAPPING
#  Maps UI category names to your chunk
#  folder names inside /chunks/
#  Update these to match YOUR actual folder
#  names inside lawyerbot_dataset/chunks/
# ══════════════════════════════════════════
CATEGORY_SOURCES = {
    "civil":        ["civil", "civil_procedure", "kerala_civil", "property", "rent", "land"],
    "criminal":     ["criminal", "ipc", "bns", "bnss", "crpc", "kerala_police", "penal"],
    "motor":        ["motor", "vehicle", "mvact", "mva", "transport"],
    "constitution": ["constitution", "fundamental", "constitutional", "directive"],
    "cyber":        ["cyber", "it_act", "information_technology", "digital", "dpdp"],
    "custom":       ["custom", "university", "college", "organisation", "rules"],
}

def get_sources_for_category(category: str) -> List[str]:
    """Return the list of source folder name keywords for a category."""
    return CATEGORY_SOURCES.get(category.lower(), [])

# ══════════════════════════════════════════
#  CHUNK TEXT READER
# ══════════════════════════════════════════
def read_chunk_text(source_file: str, chunk_file: str) -> str:
    """
    Read the actual text of a chunk from the chunks/ folder.
    Falls back gracefully if file not found.
    """
    chunk_path = CHUNKS_DIR / source_file / chunk_file
    if chunk_path.exists():
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"  ⚠ Could not read chunk {chunk_path}: {e}")
    return ""

# ══════════════════════════════════════════
#  RETRIEVAL FUNCTION
# ══════════════════════════════════════════
def retrieve_chunks(question: str, category: str, top_k: int = TOP_K) -> List[dict]:
    """
    1. Embed the user question
    2. Search FAISS for nearest vectors
    3. Filter results by category (using source_file name)
    4. Read and return the actual chunk texts
    """

    # Embed question using same model as vector.py
    question_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")

    # Search more results so we have enough after category filtering
    search_k = top_k * 6
    distances, indices = faiss_index.search(question_embedding, search_k)

    category_keywords = get_sources_for_category(category)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        source_file = meta.get("source_file", "").lower()
        chunk_file  = meta.get("chunk_file", "")

        # Filter by category — check if source_file contains any category keyword
        # If no keywords match (general query), include all results
        if category_keywords:
            if not any(kw in source_file for kw in category_keywords):
                continue

        # Read actual chunk text from chunks/ folder
        text = read_chunk_text(source_file, chunk_file)
        if not text:
            continue

        results.append({
            "text":        text,
            "source_file": source_file,
            "chunk_file":  chunk_file,
            "distance":    float(dist)
        })

        if len(results) >= top_k:
            break

    # If no category-specific results found, fall back to top general results
    if not results:
        print(f"  ⚠ No category-specific results for '{category}', using general search")
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            meta        = metadata[idx]
            source_file = meta.get("source_file", "")
            chunk_file  = meta.get("chunk_file", "")
            text        = read_chunk_text(source_file, chunk_file)
            if text:
                results.append({
                    "text":        text,
                    "source_file": source_file,
                    "chunk_file":  chunk_file,
                    "distance":    float(dist)
                })
            if len(results) >= top_k:
                break

    return results

# ══════════════════════════════════════════
#  SYSTEM PROMPTS PER CATEGORY
# ══════════════════════════════════════════
BASE_INSTRUCTION = """You are LawyerBot, an expert AI legal assistant for Kerala, India.
You answer questions using retrieved legal document excerpts provided below.
Always cite the source document and relevant section/act in your answer.
Be clear, accurate, and easy for a non-lawyer to understand.
End every response with:
"⚠ This is general legal information only. Please consult a licensed advocate for advice specific to your situation." """

SYSTEM_PROMPTS = {
    "civil": BASE_INSTRUCTION + "\nYou specialize in Kerala Civil Law: property, land registration, tenancy, rent control, civil procedures, succession, Kerala Land Reforms Act, CPC, Limitation Act.",
    "criminal": BASE_INSTRUCTION + "\nYou specialize in Kerala Criminal Law: FIR, bail, arrest, IPC/BNS offences, BNSS procedures, Kerala Police Act, trial process, anticipatory bail.",
    "motor": BASE_INSTRUCTION + "\nYou specialize in Motor Vehicle Laws for Kerala: MVA 1988, Kerala MVD rules, fines, licensing, vehicle registration, insurance, MACT claims.",
    "constitution": BASE_INSTRUCTION + "\nYou specialize in the Indian Constitution: Fundamental Rights, Directive Principles, writ petitions, PILs, landmark SC judgments, Articles 14, 19, 21, 32, 226.",
    "cyber": BASE_INSTRUCTION + "\nYou specialize in Cyber and IT Law in India: IT Act 2000, IT Amendment 2008, DPDP Act 2023, cybercrime, online fraud, data privacy, Kerala Cyber Cell.",
    "custom": BASE_INSTRUCTION + "\nYou handle custom institutional queries. Answer ONLY from the provided documents. If not found, clearly say: 'This information is not in the provided documents.'",
    "general": BASE_INSTRUCTION + "\nAnswer general Indian and Kerala legal questions across all areas of law.",
}

# ══════════════════════════════════════════
#  GROQ ANSWER GENERATION
# ══════════════════════════════════════════
def generate_answer(
    question: str,
    chunks: List[dict],
    category: str,
    chat_history: List[dict]
) -> str:
    """
    Build a prompt with retrieved legal chunks and generate
    a natural language answer using Groq Llama.
    """

    # Format retrieved legal context
    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Excerpt {i} — Source: {chunk['source_file']}]\n{chunk['text']}"
            )
        context_block = "\n\n".join(context_parts)
        context_section = f"""
══ RETRIEVED LEGAL EXCERPTS ══
{context_block}
══ END OF EXCERPTS ══

Base your answer on the above excerpts. Cite sources where relevant.
"""
    else:
        context_section = "\nNo specific documents retrieved. Answer from general knowledge of Indian and Kerala law.\n"

    # Final system prompt = category prompt + retrieved context
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["general"]) + context_section

    # Build messages: system + last 6 history messages (keeps tokens low)
    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-6:]:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # your model from rag_groq.py
        messages=messages,
        temperature=0.2,    # low = more factual for legal answers
        max_tokens=1024,
    )

    return response.choices[0].message.content

# ══════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════
app = FastAPI(
    title="LawyerBot API",
    description="Kerala Legal AI — FAISS + Groq Llama RAG Pipeline",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace * with your domain when deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    category: str            # civil/criminal/motor/constitution/cyber/custom
    messages: List[Message]  # full conversation history from frontend

class ChatResponse(BaseModel):
    reply: str
    category: str
    sources: List[str]        # which documents were used
    chunks_retrieved: int     # how many chunks were found

# ══════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    category = request.category.lower()

    if category not in SYSTEM_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Valid: civil, criminal, motor, constitution, cyber, custom"
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # Get the latest user question
    user_question = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )
    if not user_question.strip():
        raise HTTPException(status_code=400, detail="No user question found.")

    # Step 1 — Retrieve relevant chunks from FAISS
    chunks = retrieve_chunks(user_question, category)
    sources = list(set(c["source_file"] for c in chunks))

    # Step 2 — Build history (exclude last user message, passed separately)
    history = [
        {"role": m.role, "content": m.content}
        for m in request.messages[:-1]
    ]

    # Step 3 — Generate answer with Groq Llama
    try:
        reply = generate_answer(user_question, chunks, category, history)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "auth" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid Groq API key. Check your .env file.")
        elif "429" in error_msg:
            raise HTTPException(status_code=429, detail="Groq rate limit reached. Wait a moment.")
        else:
            raise HTTPException(status_code=500, detail=f"LLM error: {error_msg}")

    return ChatResponse(
        reply=reply,
        category=category,
        sources=sources,
        chunks_retrieved=len(chunks)
    )


@app.get("/")
def root():
    return {
        "app":     "LawyerBot API",
        "version": "2.0.0",
        "status":  "running",
        "vectors": faiss_index.ntotal,
        "chunks":  len(metadata),
        "tip":     "Visit /docs for interactive API testing"
    }


@app.get("/health")
def health():
    """Check if index, metadata and chunk folders are all ready."""

    # Check which source folders exist in chunks/
    chunk_folders = []
    if CHUNKS_DIR.exists():
        chunk_folders = [f.name for f in CHUNKS_DIR.iterdir() if f.is_dir()]

    # Check category coverage
    coverage = {}
    for cat, keywords in CATEGORY_SOURCES.items():
        matched = [f for f in chunk_folders if any(kw in f.lower() for kw in keywords)]
        coverage[cat] = matched if matched else "⚠ No matching folders found"

    return {
        "status":         "ok",
        "faiss_index":    str(VECTOR_DB),
        "vectors_loaded": faiss_index.ntotal,
        "chunks_loaded":  len(metadata),
        "embed_model":    "all-MiniLM-L6-v2",
        "llm_model":      "llama-3.3-70b-versatile",
        "chunk_folders":  chunk_folders,
        "category_coverage": coverage
    }


@app.get("/sources")
def list_sources():
    """List all unique source documents in the vector database."""
    sources = sorted(set(m.get("source_file", "unknown") for m in metadata))
    return {
        "total_sources": len(sources),
        "total_chunks":  len(metadata),
        "sources": sources
    }
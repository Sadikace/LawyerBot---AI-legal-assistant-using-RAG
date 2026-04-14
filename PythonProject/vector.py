import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_BASE = r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\chunks"
DB_PATH = r"C:\Users\sadik\OneDrive\Desktop\lawyerbot_dataset\vectordb"

os.makedirs(DB_PATH, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metadata = []

print("📂 Reading chunk files...")

for folder in os.listdir(CHUNKS_BASE):
    folder_path = os.path.join(CHUNKS_BASE, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                texts.append(content)
                metadata.append({
                    "source_file": folder,
                    "chunk_file": file
                })

print(f"🔢 Total chunks: {len(texts)}")

print("🧠 Creating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(DB_PATH, "lawyerbot.index"))

with open(os.path.join(DB_PATH, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("✅ Vector DB created successfully!")
print("📁 Saved in:", DB_PATH)

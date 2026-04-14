

#  LawyerBot - AI legal assistant using RAG

> An AI-powered legal chatbot for Kerala \\\\\\\& Indian law, built using Retrieval Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-purple?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-yellow?style=flat-square)
---

##  About

LawyerBot is an AI-powered legal assistant that answers questions about Kerala and Indian law using a **Retrieval Augmented Generation (RAG)** pipeline. It retrieves relevant sections from real legal documents and generates accurate, context-aware answers using **Llama 3.3 70B** via the Groq API.

it retrieves relevent sections from legal documents and generates human understanding language

---

##  Features

-  **Law Categories** — Civil, Criminal, Motor Vehicles, Constitution, Cyber IT Law, Custom Integration and  can be increase depending on developers choice
-  **RAG Pipeline** — FAISS vector search retrieves exact law sections before generating answers
-  **Ultra-fast responses** — Powered by Groq API (fastest LLM inference available)
-  **Custom Integration** — Universities, colleges, and organizations can plug in their own rules and policies
-  **Chat History** — Conversational context with last 6 messages passed to the LLM
-  **Export Chat** — Save legal Q&A sessions as `.txt` files
-  **Source Citations** — Every answer shows which documents were used

---

##  System Architecture

```

User Question

\\\&#x20;     ↓

FastAPI (api.py)          ← receives HTTP request

\\\&#x20;     ↓

all-MiniLM-L6-v2          ← converts question to 384-dim vector

\\\&#x20;     ↓

FAISS (lawyerbot.index)   ← finds top 5 most similar law chunks

\\\&#x20;     ↓

Groq API + Llama 3.3 70B  ← generates answer using retrieved chunks

\\\&#x20;     ↓

FastAPI                   ← sends JSON response back

\\\&#x20;     ↓

lawyerbot.html            ← displays answer to user

```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Backend** | FastAPI + Uvicorn (Python 3.11) |
| **LLM** | Llama 3.3 70B via Groq API |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Vector DB** | FAISS (IndexFlatL2) |
| **PDF Extraction** | pdfplumber |
| **Environment** | python-dotenv |

---

##  Project Structure

```

lawyerbot\\\\\\\_dataset/

├── raw\\\\\\\_sources/          ← original PDF law documents

├── extracted\\\\\\\_text/       ← raw extracted text from PDFs

├── cleaned\\\\\\\_text/         ← cleaned and normalized text

├── chunks/               ← chunked text files (500 words, 80 overlap)

│   ├── civil\\\\\\\_law/

│   ├── criminal\\\\\\\_law/

│   └── ...

├── vectordb/

│   ├── lawyerbot.index   ← FAISS vector index

│   └── metadata.json     ← chunk-to-source mapping

└── PythonProject/ 

https://github.com/user-attachments/assets/30759d33-62c2-4502-ac5d-fe24cf709a7b



\\\&#x20;   ├── api.py            ← main FastAPI backend

\\\&#x20;   ├── charecter.py      ← text cleaning script

\\\&#x20;   ├── chunk.py          ← text chunking script

\\\&#x20;   ├── vector.py         ← FAISS index builder

\\\&#x20;   ├── lawyerbot.html    ← main chat UI

\\\&#x20;   |── testing.py        ← for testing RAG

\\\&#x20;   ├── .env              ← API keys (not uploaded to GitHub)

\\\&#x20;   └── requirements.txt

```

---

##  RAG Configuration

| Parameter | Value |
|---|---|
| Chunk Size | 500 words |
| Chunk Overlap | 80 words |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dimensions | 384 |
| FAISS Index Type | IndexFlatL2 |
| Top-K Retrieval | 5 chunks |
| LLM Temperature | 0.2 |
| Max Tokens | 1024 |
| History Window | Last 6 messages |

---

##  Law Categories Covered

| Category | Acts & Sources |
|---|---|
|  Civil Law | CPC 1908, Kerala Land Reforms Act 1963, Kerala Rent Control Act 1965, Limitation Act 1963 |
|  Criminal Law | BNS 2023, BNSS 2023, IPC, Kerala Police Act 2011, Prevention of Corruption Act |
|  Motor Vehicles | Motor Vehicles Act 1988, Kerala MVD Rules 1989, MACT, Third Party Insurance |
|  Constitution | Fundamental Rights (Part III), Directive Principles (Part IV), Writs, PIL |
|  Cyber & IT Law | IT Act 2000, ITAA 2008, DPDP Act 2023, Kerala Cyber Cell |
|  Custom Integration | University rules, college policies, organizational regulations |

---

##  Getting Started

### Prerequisites
- Python 3.11
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash

\\\\# 1. Clone the repository

git clone https://github.com/YOUR\\\\\\\_USERNAME/lawyerbot.git

cd lawyerbot/PythonProject



\\\\# 2. Create and activate virtual environment

python -m venv .venv

.venv\\\\\\\\Scripts\\\\\\\\activate        # Windows

\\\\# source .venv/bin/activate   # Mac/Linux



\\\\# 3. Install dependencies

pip install -r requirements.txt



\\\\# 4. Create .env file

echo GROQ\\\\\\\_API\\\\\\\_KEY=your\\\\\\\_groq\\\\\\\_key\\\\\\\_here > .env



\\\\# 5. Run the server

uvicorn api:app --reload

```


## Requirements

```

fastapi

uvicorn\\\\\\\[standard]

groq

sentence-transformers

faiss-cpu

numpy

python-dotenv

pdfplumber

python-jose\\\\\\\[cryptography]

passlib\\\\\\\[bcrypt]

tf-keras

```

---

##  Adding New PDF Documents

```bash

\\\\# 1. Add PDFs to raw\\\\\\\_sources/ folder

\\\\# 2. Extract text

python first.py



\\\\# 3. Clean text

python charecter.py



\\\\# 4. Chunk text

python chunk.py



\\\\# 5. Rebuild FAISS index

python vector.py



\\\\# 6. Restart server

uvicorn api:app --reload

```
## Demo


https://github.com/user-attachments/assets/7b8f52a2-fcf3-42cb-84bc-8b47b33251c1



---
## Author


| **Muhammed Sadik** |

---

##  Disclaimer

LawyerBot provides **general legal information only**. It is not a substitute for professional legal advice from a licensed advocate. Always consult a qualified lawyer for your specific legal situation.

---

##  License

this project is licensed under the MIT License.


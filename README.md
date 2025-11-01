# Graph-Powered Recommendation & Reasoning Engine

**Explainable Retrieval-Augmented Generation (RAG) system combining Graph Intelligence, Vector Search, and LLM reasoning.**

This project demonstrates how **graph-structured knowledge** and **vector embeddings** can work together to power *explainable AI* systems, providing grounded answers, interpretable reasoning paths, and meaningful recommendations.

---

## Overview

This MVP implements a hybrid **Graph + Vector + RAG** architecture:

- **Knowledge Graph** built from documents, chunks, and extracted concepts  
- **Vector Index (FAISS)** for dense semantic retrieval  
- **Graph Expansion** for contextual reasoning across related concepts  
- **RAG Engine** for answering questions with citations and reasoning paths  
- **Recommendation Engine** combining vector similarity and PageRank  
- **FastAPI Backend** and **Streamlit Frontend** for interactive use  

It’s fully extensible, swap the in-memory graph for **Neo4j**, or replace extractive responses with **LLM-generated answers** via OpenAI or Hugging Face.

---

## Project Structure

```
graph-rag-engine/
│
├── ingest/             # Document ingestion & embedding
│   ├── split.py
│   └── ingest_docs.py
│
├── graph/              # Knowledge graph (NetworkX)
│   └── graph_store.py
│
├── backend/            # FastAPI service & retriever
│   ├── api.py
│   ├── retriever.py
│   └── rag.py
│
├── ui/                 # Streamlit interface
│   └── app.py
│
├── data/               # Sample documents & index output
├── env/                # Dependencies
└── README.md
```

---

## Features

| Category | Description |
|-----------|--------------|
| **Ingestion** | Automatic document parsing, chunking, and concept extraction |
| **Graph Layer** | Links Docs ↔ Chunks ↔ Concepts with PageRank weighting |
| **Vector Layer** | FAISS similarity search over sentence embeddings |
| **Hybrid Retrieval** | Combines vector similarity + concept overlap + PageRank |
| **Explainability** | Returns graph paths showing *why* each answer was chosen |
| **Recommendations** | Suggests related documents via hybrid graph + vector scoring |
| **UI/UX** | Streamlit web app for QA & recommendation exploration |
| **API** | FastAPI endpoints for `/ask`, `/recommend`, and `/docs_list` |

---

## Architecture

```
┌────────────┐       ┌────────────┐        ┌───────────────┐
│  Documents │ ───▶  │ Chunking & │ ───▶  │   Embeddings  │
│  (Markdown)│       │ Concept NER│        │ (FAISS Index) │
└────────────┘       └────────────┘        └──────┬────────┘
                                                  │
                                      ┌───────────▼───────────┐
                                      │   Knowledge Graph     │
                                      │ (Docs–Chunks–Concepts)│
                                      └───────────┬───────────┘
                                                  │
                              ┌───────────────────▼───────────────────┐
                              │  Hybrid Retriever + RAG Composition   │
                              │   (Vector + Graph Expansion + PR)     │
                              └───────────────────┬───────────────────┘
                                                  │
                                 ┌────────────────▼─────────────────┐
                                 │  FastAPI  +  Streamlit  Frontend │
                                 └──────────────────────────────────┘
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your_username>/graph-rag-engine.git
cd graph-rag-engine

# 2. (Optional) create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r env/requirements.txt
```

---

## Usage

### Step 1 – Ingest Documents

```bash
python -m ingest.ingest_docs
```

This:
- Reads documents from `data/docs/`
- Splits them into semantic chunks  
- Extracts key concepts  
- Builds FAISS vector index and in-memory graph  

### Step 2 – Run Backend API

```bash
uvicorn backend.api:app --reload --port 8000
```

API runs at: [http://localhost:8000](http://localhost:8000)

### Step 3 – Launch Streamlit UI

```bash
streamlit run ui/app.py
```

UI runs at: [http://localhost:8501](http://localhost:8501)

---

## Example Queries

- “What is FAISS and how does it relate to embeddings?”
- “Explain how TF-IDF differs from dense embeddings.”
- “Find documents related to Streamlit.”

Each response includes:
- **Answer text**  
- **Citations** (titles + URLs)  
- **Graph paths** explaining *why those sources were chosen*  

---

## API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/health` | GET | Health check |
| `/ask` | POST | `{ "question": "..." }` → QA response |
| `/recommend` | POST | `{ "doc_id": "..." }` → related docs |
| `/docs_list` | GET | Returns all indexed documents |

---

## Tech Stack

- **Language:** Python 3.10 +
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **Graph Layer:** NetworkX (Neo4j-ready)
- **Data Handling:** NumPy / Pandas
- **ML Ops Ready:** Easily extendable with MLflow, Airflow, etc.

---

## Extending the Project

| Upgrade | Description |
|----------|--------------|
| **LLM Integration** | Replace `compose_answer_extractive()` with GPT-4 or Llama 2 for abstractive answers |
| **Neo4j Graph DB** | Swap in `graph/neo4j_store.py` to persist graph and visualize it |
| **Vector DB** | Replace FAISS with Pinecone, Qdrant, or Weaviate |
| **GNN Models** | Use GraphSAGE or GAT for better document representations |
| **MLOps** | Add monitoring, retraining triggers, or CI/CD pipelines |

---

## Roadmap

- [ ] Integrate GPT-4 / Llama 2 for abstractive RAG answers  
- [ ] Add Neo4j visualization layer  
- [ ] Build GNN-based recommender  
- [ ] Introduce feedback-based re-ranking  
- [ ] Deploy Dockerized demo to Hugging Face Spaces  

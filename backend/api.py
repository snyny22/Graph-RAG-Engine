from fastapi import FastAPI
from pydantic import BaseModel
from .rag import ask
from .retriever import recommend_similar, CHUNKS
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Graph RAG MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    question: str

class RecReq(BaseModel):
    doc_id: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask_ep(payload: AskReq):
    return ask(payload.question)

@app.post("/recommend")
def rec_ep(payload: RecReq):
    return {"items": recommend_similar(payload.doc_id)}

@app.get("/docs_list")
def docs_list():
    docs = {}
    for c in CHUNKS:
        docs[c["doc_id"]] = {"title": c["doc_title"], "url": c["url"]}
    out = [{"doc_id": k, **v} for k,v in docs.items()]
    return {"items": out}

import os, glob, pickle, orjson
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from .split import simple_chunk, extract_concepts
from pathlib import Path
from graph.graph_store import GraphStore

BASE = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE / "data" / "docs"
OUT_DIR = BASE / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_docs() -> List[Dict[str, Any]]:
    docs = []
    for path in glob.glob(str(DOCS_DIR / "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "id": os.path.splitext(os.path.basename(path))[0],
            "title": os.path.basename(path),
            "url": f"file://{path}",
            "text": text
        })
    return docs

def build_index(docs: List[Dict[str, Any]]):
    # chunk + concepts
    chunks = []
    concepts = set()
    for d in docs:
        for i, ch in enumerate(simple_chunk(d["text"])):
            cid = f"{d['id']}_chunk_{i}"
            cpts = extract_concepts(ch)
            concepts.update(cpts)
            chunks.append({
                "id": cid,
                "doc_id": d["id"],
                "doc_title": d["title"],
                "url": d["url"],
                "text": ch,
                "concepts": cpts
            })
    # embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype(np.float32))

    # persist
    with open(OUT_DIR / "docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    with open(OUT_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(OUT_DIR / "vectors.npy", vecs.astype(np.float32))
    faiss.write_index(index, str(OUT_DIR / "faiss.index"))

    # build graph
    gs = GraphStore()
    for d in docs:
        gs.add_doc(d["id"], d["title"], d["url"])
    for c in chunks:
        gs.add_chunk(c["id"], c["text"], c["doc_id"])
        for k in c["concepts"]:
            gs.add_concept(k)
            gs.link_mentions(c["id"], k)
    gs.compute_doc_pagerank()
    gs.save(OUT_DIR / "graph.pkl")

if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        raise SystemExit(f"No docs found in {DOCS_DIR}")
    build_index(docs)
    print(f"Ingested {len(docs)} docs. Index written to {OUT_DIR}")

from pathlib import Path
import pickle, faiss, numpy as np, orjson
from sentence_transformers import SentenceTransformer
from graph.graph_store import GraphStore

BASE = Path(__file__).resolve().parents[1]
IDX_DIR = BASE / "data" / "index"

with open(IDX_DIR / "chunks.pkl", "rb") as f:
    CHUNKS = pickle.load(f)
VECS = np.load(IDX_DIR / "vectors.npy")
INDEX = faiss.read_index(str(IDX_DIR / "faiss.index"))
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
GRAPH = GraphStore.load(IDX_DIR / "graph.pkl")

CHUNK_BY_ID = {c["id"]: c for c in CHUNKS}

def ann_search(q: str, k: int = 8):
    qv = MODEL.encode([q], normalize_embeddings=True).astype(np.float32)
    D, I = INDEX.search(qv, k)
    ids = [CHUNKS[i]["id"] for i in I[0]]
    sims = [float(D[0][j]) for j in range(len(ids))]
    return list(zip(ids, sims))

def expand_and_rerank(q: str, base_k: int = 8, expand_hops: int = 1, top_n: int = 6):
    base = ann_search(q, k=base_k)
    candidate_ids = set(cid for cid,_ in base)
    # expand via shared concepts
    for cid,_ in base:
        neighs = GRAPH.neighbor_chunks_by_concepts(cid, max_neighbors=6)
        candidate_ids.update(neighs)

    qv = MODEL.encode([q], normalize_embeddings=True)[0]
    scored = []
    for cid in candidate_ids:
        c = CHUNK_BY_ID[cid]
        # embedding sim using precomputed vecs
        idx = next(i for i,x in enumerate(CHUNKS) if x["id"]==cid)
        emb_sim = float(np.dot(qv, VECS[idx]))
        # concept overlap
        q_terms = set(t for t in q.lower().split() if len(t)>2)
        c_overlap = len(q_terms.intersection(set(c["concepts"])))
        # doc pagerank
        doc_id = c["doc_id"]
        doc_pr = GRAPH.get_doc_info(doc_id).get("pagerank", 0.0)
        score = 0.6*emb_sim + 0.25*c_overlap + 0.15*doc_pr
        scored.append((score, cid))
    scored.sort(reverse=True)
    top = [cid for _, cid in scored[:top_n]]
    return [CHUNK_BY_ID[cid] for cid in top]

def recommend_similar(doc_id: str, k: int = 5):
    # compute centroid of doc's chunks and find nearest other docs
    doc_chunks = [c for c in CHUNKS if c["doc_id"]==doc_id]
    if not doc_chunks: return []
    idxs = [i for i,c in enumerate(CHUNKS) if c["doc_id"]==doc_id]
    centroid = VECS[idxs].mean(axis=0).astype(np.float32)
    D, I = INDEX.search(centroid.reshape(1,-1), 32)
    doc_scores = {}
    for j in I[0]:
        d = CHUNKS[j]["doc_id"]
        if d != doc_id:
            doc_scores[d] = max(doc_scores.get(d, 0.0), float(np.dot(centroid, VECS[j])))
    # blend with pagerank
    for d in list(doc_scores.keys()):
        doc_scores[d] = 0.8*doc_scores[d] + 0.2*GRAPH.get_doc_info(d).get("pagerank", 0.0)
    return sorted([{"doc_id":k, **GRAPH.get_doc_info(k), "score":v} for k,v in doc_scores.items()], key=lambda x:-x["score"])[:k]

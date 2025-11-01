from typing import List, Dict, Any
from .retriever import expand_and_rerank, GRAPH

def compose_answer_extractive(question: str, passages: List[Dict[str, Any]]) -> str:
    # Simple extractive 'answer': concatenate top passages with citations.
    # This is placeholder for an LLM call (can replace later).
    parts = []
    for p in passages:
        parts.append(f"**Source:** [{p['doc_title']}]({p['url']})\n> {p['text']}")
    return "\n\n".join(parts)

def ask(question: str):
    passages = expand_and_rerank(question, base_k=8, expand_hops=1, top_n=5)
    answer = compose_answer_extractive(question, passages)
    chunk_ids = [p["id"] for p in passages]
    paths = GRAPH.explain_paths(chunk_ids)
    citations = [{"doc_title": p["doc_title"], "url": p["url"]} for p in passages]
    return {
        "answer": answer,
        "citations": citations,
        "paths": paths
    }

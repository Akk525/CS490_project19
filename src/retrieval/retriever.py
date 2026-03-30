from typing import Dict, List, Tuple

import numpy as np

from src.retrieval.embedder_base import EmbedderBase


def top_k_by_cosine(query_embedding: np.ndarray, matrix: np.ndarray, k: int) -> List[Tuple[int, float]]:
    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    scores = m @ q
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def retrieve(query: str, embedder: EmbedderBase, index_matrix: np.ndarray, library_rows: List[Dict], k: int) -> List[Dict]:
    q_emb = np.array(embedder.embed([query])[0], dtype=np.float32)
    ids = top_k_by_cosine(q_emb, index_matrix, k)
    out = []
    for rank, (i, score) in enumerate(ids):
        row = dict(library_rows[i])
        row['rank'] = rank + 1
        row['retrieval_score'] = score
        out.append(row)
    return out

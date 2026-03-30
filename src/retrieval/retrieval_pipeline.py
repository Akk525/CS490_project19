from typing import Dict, List, Optional

import numpy as np

from src.dataset.schema import ProceduralExample
from src.retrieval.query_builder import build_query
from src.retrieval.retriever import retrieve


def run_retrieval(
    example: ProceduralExample,
    strategy: str,
    k: int,
    candidate_pool: int,
    embedder,
    index_matrix: np.ndarray,
    library_rows: List[Dict],
    reranker: Optional[object] = None,
) -> Dict:
    query = build_query(example, strategy)
    candidates = retrieve(query, embedder, index_matrix, library_rows, candidate_pool)
    before = [{'example_id': x['example_id'], 'rank': x['rank'], 'score': x.get('retrieval_score')} for x in candidates]

    if reranker:
        docs = [f"{c['goal']} | {' | '.join(c['full_procedure'])} | {c['disruption_description']}" for c in candidates]
        reranked = reranker.rerank(query, docs)
        reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)[:k]
        after = []
        selected = []
        for new_rank, (idx, score) in enumerate(reranked_sorted, start=1):
            c = candidates[idx]
            selected.append(c)
            after.append({'example_id': c['example_id'], 'rank': new_rank, 'score': score})
    else:
        selected = candidates[:k]
        after = [{'example_id': c['example_id'], 'rank': c['rank'], 'score': c.get('retrieval_score')} for c in selected]

    return {
        'query_text': query,
        'retrieved_candidates_before_rerank': before,
        'retrieved_candidates_after_rerank': after,
        'retrieval_strategy': strategy,
        'selected_examples': selected,
    }

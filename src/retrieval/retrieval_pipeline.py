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
    rerank_fallback_to_retrieval: bool,
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
        try:
            reranked = reranker.rerank(query, docs)
        except Exception as exc:
            doc_lengths = [len(doc) for doc in docs]
            if rerank_fallback_to_retrieval:
                selected = candidates[:k]
                after = [
                    {'example_id': c['example_id'], 'rank': c['rank'], 'score': c.get('retrieval_score')}
                    for c in selected
                ]
                return {
                    'query_text': query,
                    'retrieved_candidates_before_rerank': before,
                    'retrieved_candidates_after_rerank': after,
                    'retrieval_strategy': strategy,
                    'selected_examples': selected,
                    'rerank_fallback_used': True,
                    'rerank_fallback_error': (
                        "Rerank failed for "
                        f"example_id={example.example_id} "
                        f"strategy={strategy} "
                        f"candidate_pool={candidate_pool} "
                        f"query_len={len(query)} "
                        f"doc_count={len(docs)} "
                        f"max_doc_len={max(doc_lengths) if doc_lengths else 0} "
                        f"avg_doc_len={sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0:.1f}"
                    ),
                }
            raise RuntimeError(
                "Rerank failed for "
                f"example_id={example.example_id} "
                f"strategy={strategy} "
                f"candidate_pool={candidate_pool} "
                f"query_len={len(query)} "
                f"doc_count={len(docs)} "
                f"max_doc_len={max(doc_lengths) if doc_lengths else 0} "
                f"avg_doc_len={sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0:.1f}"
            ) from exc
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
        'rerank_fallback_used': False,
    }

from typing import Any, Dict, List


def _first_present(payload: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def normalize_generation_text(payload: Dict[str, Any]) -> str:
    # OpenAI-compatible
    if isinstance(payload.get('choices'), list) and payload['choices']:
        msg = payload['choices'][0].get('message', {})
        if isinstance(msg, dict) and msg.get('content'):
            return str(msg['content'])
        txt = payload['choices'][0].get('text')
        if txt:
            return str(txt)

    # Common HTTP wrappers
    txt = _first_present(payload, ['text', 'output_text', 'response', 'generated_text'])
    if txt is not None:
        return str(txt)

    # Tool-style payloads
    data = payload.get('data')
    if isinstance(data, dict):
        nested = _first_present(data, ['text', 'output_text', 'response'])
        if nested is not None:
            return str(nested)

    raise ValueError('Could not normalize generation response. Expected one of choices/message/content or text/output_text/response.')


def normalize_embeddings(payload: Dict[str, Any]) -> List[List[float]]:
    if isinstance(payload.get('data'), list):
        first = payload['data'][0] if payload['data'] else None
        if isinstance(first, dict) and 'embedding' in first:
            return [row['embedding'] for row in payload['data']]

    embs = _first_present(payload, ['embeddings', 'vectors'])
    if isinstance(embs, list):
        return embs

    data = payload.get('data')
    if isinstance(data, dict):
        embs = _first_present(data, ['embeddings', 'vectors'])
        if isinstance(embs, list):
            return embs

    raise ValueError('Could not normalize embedding response. Expected data[].embedding or embeddings/vectors.')


def normalize_rerank_results(payload: Dict[str, Any]) -> List[Dict[str, float]]:
    results = _first_present(payload, ['results', 'reranked'])
    if not isinstance(results, list):
        data = payload.get('data')
        if isinstance(data, dict):
            results = _first_present(data, ['results', 'reranked'])
    if not isinstance(results, list):
        raise ValueError('Could not normalize reranker response. Expected results list.')

    normalized = []
    for i, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        idx = row.get('index', row.get('idx', i))
        score = row.get('score', row.get('relevance_score', 0.0))
        normalized.append({'index': int(idx), 'score': float(score)})
    return normalized

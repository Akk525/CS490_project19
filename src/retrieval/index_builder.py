from pathlib import Path
from typing import Dict, List

import numpy as np

from src.retrieval.embedder_base import EmbedderBase
from src.utils.io import write_jsonl


def _index_text(row: Dict) -> str:
    metadata = row.get('metadata') or {}
    missing = metadata.get('missing_ingredient')
    substitute = metadata.get('suggested_substitute')
    lines = [
        f"Goal: {row['goal']}",
        f"Procedure: {' | '.join(row['full_procedure'])}",
        f"Disrupted step: {row.get('disrupted_step_text', '')}",
        f"Disruption type: {row.get('disruption_type', '')}",
        f"Disruption: {row.get('disruption_description', '')}",
    ]
    if missing:
        lines.append(f"Missing ingredient: {missing}")
    if substitute:
        lines.append(f"Known safe substitute: {substitute}")
    if row.get('target_adaptation'):
        lines.append(f"Adaptation: {row['target_adaptation']}")
    return "\n".join(lines)


def build_index(rows: List[Dict], embedder: EmbedderBase, out_index: Path, out_library: Path) -> None:
    texts = [_index_text(r) for r in rows]
    batch_size = 8
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            all_embeddings.extend(embedder.embed(batch))
        except Exception:
            if len(batch) == 1:
                raise RuntimeError(f"Embedding failed for retrieval row index {start}") from None

            for offset, text in enumerate(batch):
                row_index = start + offset
                try:
                    all_embeddings.extend(embedder.embed([text]))
                except Exception:
                    raise RuntimeError(
                        f"Embedding failed for retrieval row index {row_index}"
                    ) from None

    embeddings = np.array(all_embeddings, dtype=np.float32)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_index, embeddings=embeddings)
    write_jsonl(out_library, rows)

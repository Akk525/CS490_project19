from pathlib import Path
from typing import Dict, List

import numpy as np

from src.retrieval.embedder_base import EmbedderBase
from src.utils.io import write_jsonl


def build_index(rows: List[Dict], embedder: EmbedderBase, out_index: Path, out_library: Path) -> None:
    texts = [f"Goal: {r['goal']}\nProcedure: {' | '.join(r['full_procedure'])}\nDisruption: {r['disruption_description']}" for r in rows]
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

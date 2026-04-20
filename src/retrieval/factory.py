from typing import Dict, Optional

from src.retrieval.hf_embedder import HuggingFaceEmbedder
from src.retrieval.hf_reranker import HuggingFaceReranker
from src.retrieval.school_server_embedder import SchoolServerEmbedder
from src.retrieval.school_server_reranker import SchoolServerReranker


def build_embedder(backend_cfg: Dict, model_name: str):
    mode = backend_cfg.get('mode')
    if mode == 'hf_local':
        return HuggingFaceEmbedder(backend_cfg, model_name)
    return SchoolServerEmbedder(backend_cfg, model_name)


def build_reranker(backend_cfg: Dict, model_name: Optional[str]):
    if not model_name:
        return None
    mode = backend_cfg.get('mode')
    if mode == 'hf_local':
        return HuggingFaceReranker(backend_cfg, model_name)
    return SchoolServerReranker(backend_cfg, model_name)

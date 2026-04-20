GENERATION_MODELS = {
    'gemma3:4b-it-q8_0',
    'olmo-3:32b-think-q8_0',
    'nemotron-3-nano:30b-a3b-q8_0',
    'gemma3:27b-it-q8_0',
    'gemma3:27b',
    'qwen3:30b-a3b-thinking-2507-q4_K_M',
    'llama3.3:70b-instruct-q8_0',
    'mistral-large:123b',
    'qwen3:235b-a22b',
    'command-a:111b',
    'gpt-oss:120b',
}

EMBEDDING_MODELS = {
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'qwen3-embedding:8b-fp16',
    'qwen3-embedding:8b-q8_0',
    'qwen3-embedding:4b-q8_0',
    'qwen3-embedding:0.6b',
    'qwen3-embedding:4b',
    'qwen3-embedding:8b',
    'nomic-embed-text:latest',
}

RERANKER_MODELS = {
    'BAAI/bge-reranker-base',
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'hf.co/jinaai/jina-reranker-v3-GGUF:BF16',
    'dengcao/Qwen3-Reranker-0.6B:F16',
    'hf.co/mradermacher/colbertv2.0-GGUF:F16',
}


def validate_model_names(generation: str, embedding: str, reranker: str) -> None:
    if not _is_supported_generation_model(generation):
        raise ValueError(f'Unsupported generation model: {generation}')
    if embedding and embedding not in EMBEDDING_MODELS:
        raise ValueError(f'Unsupported embedding model: {embedding}')
    if reranker and reranker not in RERANKER_MODELS:
        raise ValueError(f'Unsupported reranker model: {reranker}')


def _is_supported_generation_model(model_name: str) -> bool:
    if model_name in GENERATION_MODELS:
        return True
    # vLLM resolves Hugging Face model ids directly, e.g. Qwen/Qwen2.5-VL-7B-Instruct.
    # Keep a small guard so accidental free-text names still fail early.
    return "/" in model_name and ":" not in model_name and len(model_name.split("/", 1)[0]) > 0

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
    'qwen3-embedding:8b-fp16',
    'qwen3-embedding:8b-q8_0',
    'qwen3-embedding:4b-q8_0',
    'qwen3-embedding:0.6b',
    'qwen3-embedding:4b',
    'qwen3-embedding:8b',
    'nomic-embed-text:latest',
}

RERANKER_MODELS = {
    'hf.co/jinaai/jina-reranker-v3-GGUF:BF16',
    'dengcao/Qwen3-Reranker-0.6B:F16',
    'hf.co/mradermacher/colbertv2.0-GGUF:F16',
}


def validate_model_names(generation: str, embedding: str, reranker: str) -> None:
    if generation not in GENERATION_MODELS:
        raise ValueError(f'Unsupported generation model: {generation}')
    if embedding not in EMBEDDING_MODELS:
        raise ValueError(f'Unsupported embedding model: {embedding}')
    if reranker not in RERANKER_MODELS:
        raise ValueError(f'Unsupported reranker model: {reranker}')

import json
import os
import subprocess
from typing import Dict, List, Tuple

import requests

from src.models.response_normalizer import normalize_rerank_results
from src.retrieval.reranker_base import RerankerBase
from src.utils.http_retry import post_with_retries


class SchoolServerReranker(RerankerBase):
    def __init__(self, backend_cfg: Dict, model_name: str):
        self.backend_cfg = backend_cfg
        self.model_name = model_name
        self.mode = backend_cfg['mode']

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        if self.mode == 'http':
            cfg = self.backend_cfg['http']
            base_url = os.getenv(cfg['base_url_env'])
            url = base_url.rstrip('/') + cfg.get('endpoint', '/rerank')
            headers = {'Content-Type': 'application/json'}
            auth_header = os.getenv(cfg.get('auth_header_env', ''), '')
            auth_token = os.getenv(cfg.get('auth_token_env', ''), '')
            if auth_header and auth_token:
                headers[auth_header] = auth_token
            payload = {'model': self.model_name, 'query': query, 'documents': documents}
            resp = post_with_retries(
                url,
                headers=headers,
                json=payload,
                timeout=self.backend_cfg.get('timeout_seconds', 120),
            )
            scored = normalize_rerank_results(resp.json())
            return [(int(x['index']), float(x['score'])) for x in scored]

        if self.mode == 'openai_compatible':
            cfg = self.backend_cfg['openai_compatible']
            base_url = os.getenv(cfg['base_url_env'])
            url = base_url.rstrip('/') + cfg.get('endpoint_path', '/rerank')
            headers = {'Content-Type': 'application/json'}
            key = os.getenv(cfg.get('api_key_env', ''))
            if key:
                headers['Authorization'] = f'Bearer {key}'
            payload = {'model': self.model_name, 'query': query, 'documents': documents}
            resp = post_with_retries(
                url,
                headers=headers,
                json=payload,
                timeout=self.backend_cfg.get('timeout_seconds', 120),
            )
            scored = normalize_rerank_results(resp.json())
            return [(int(x['index']), float(x['score'])) for x in scored]

        if self.mode == 'cli':
            cfg = self.backend_cfg['cli']
            bin_cmd = os.getenv(cfg['bin_env'], 'python')
            proc = subprocess.run(
                [bin_cmd, '-m', 'school_server_rerank', '--model', self.model_name],
                input=json.dumps({'query': query, 'documents': documents}).encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.getenv(cfg['timeout_seconds_env'], '120')),
                check=True,
            )
            parsed = json.loads(proc.stdout.decode('utf-8'))
            return [(int(x['index']), float(x['score'])) for x in parsed['results']]

        raise ValueError(f'Unsupported reranker backend mode: {self.mode}')

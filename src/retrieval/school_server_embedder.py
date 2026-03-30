import json
import os
import subprocess
from typing import Dict, List

import requests

from src.models.response_normalizer import normalize_embeddings
from src.retrieval.embedder_base import EmbedderBase


class SchoolServerEmbedder(EmbedderBase):
    def __init__(self, backend_cfg: Dict, model_name: str):
        self.backend_cfg = backend_cfg
        self.model_name = model_name
        self.mode = backend_cfg['mode']

    def _headers(self, cfg: Dict) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        key_env = cfg.get('api_key_env')
        if key_env and os.getenv(key_env):
            headers['Authorization'] = f'Bearer {os.getenv(key_env)}'
        return headers

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.mode == 'openai_compatible':
            cfg = self.backend_cfg['openai_compatible']
            base_url = os.getenv(cfg['base_url_env'])
            if not base_url:
                raise ValueError(f'Missing env: {cfg["base_url_env"]}')
            url = base_url.rstrip('/') + '/embeddings'
            payload = {'model': self.model_name, 'input': texts}
            resp = requests.post(url, headers=self._headers(cfg), json=payload, timeout=self.backend_cfg.get('timeout_seconds', 120))
            resp.raise_for_status()
            return normalize_embeddings(resp.json())

        if self.mode == 'http':
            cfg = self.backend_cfg['http']
            base_url = os.getenv(cfg['base_url_env'])
            url = base_url.rstrip('/') + cfg.get('endpoint', '/embed')
            headers = {'Content-Type': 'application/json'}
            auth_header = os.getenv(cfg.get('auth_header_env', ''), '')
            auth_token = os.getenv(cfg.get('auth_token_env', ''), '')
            if auth_header and auth_token:
                headers[auth_header] = auth_token
            payload = {'model': self.model_name, 'texts': texts}
            resp = requests.post(url, headers=headers, json=payload, timeout=self.backend_cfg.get('timeout_seconds', 120))
            resp.raise_for_status()
            return normalize_embeddings(resp.json())

        if self.mode == 'cli':
            cfg = self.backend_cfg['cli']
            bin_cmd = os.getenv(cfg['bin_env'], 'python')
            proc = subprocess.run(
                [bin_cmd, '-m', 'school_server_embed', '--model', self.model_name],
                input=json.dumps({'texts': texts}).encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.getenv(cfg['timeout_seconds_env'], '120')),
                check=True,
            )
            return json.loads(proc.stdout.decode('utf-8'))['embeddings']

        raise ValueError(f'Unsupported embedding backend mode: {self.mode}')

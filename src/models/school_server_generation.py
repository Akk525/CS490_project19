import json
import os
import subprocess
from typing import Dict

import requests

from src.models.generation_base import GenerationBase
from src.models.response_normalizer import normalize_generation_text


class SchoolServerGeneration(GenerationBase):
    def __init__(self, backend_cfg: Dict, model_name: str):
        self.backend_cfg = backend_cfg
        self.model_name = model_name
        self.mode = backend_cfg['mode']

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.mode == 'openai_compatible':
            cfg = self.backend_cfg['openai_compatible']
            base_url = os.getenv(cfg['base_url_env'])
            if not base_url:
                raise ValueError(f'Missing env: {cfg["base_url_env"]}')
            url = base_url.rstrip('/') + '/chat/completions'
            headers = {'Content-Type': 'application/json'}
            key = os.getenv(cfg.get('api_key_env', ''))
            if key:
                headers['Authorization'] = f'Bearer {key}'
            payload = {
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=self.backend_cfg.get('timeout_seconds', 180))
            resp.raise_for_status()
            return normalize_generation_text(resp.json())

        if self.mode == 'http':
            cfg = self.backend_cfg['http']
            base_url = os.getenv(cfg['base_url_env'])
            url = base_url.rstrip('/') + cfg.get('endpoint', '/generate')
            headers = {'Content-Type': 'application/json'}
            auth_header = os.getenv(cfg.get('auth_header_env', ''), '')
            auth_token = os.getenv(cfg.get('auth_token_env', ''), '')
            if auth_header and auth_token:
                headers[auth_header] = auth_token
            payload = {'model': self.model_name, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature}
            resp = requests.post(url, headers=headers, json=payload, timeout=self.backend_cfg.get('timeout_seconds', 180))
            resp.raise_for_status()
            return normalize_generation_text(resp.json())

        if self.mode == 'cli':
            cfg = self.backend_cfg['cli']
            bin_cmd = os.getenv(cfg['bin_env'], 'python')
            proc = subprocess.run(
                [bin_cmd, '-m', 'school_server_generate', '--model', self.model_name],
                input=json.dumps({'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature}).encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.getenv(cfg['timeout_seconds_env'], '180')),
                check=True,
            )
            return json.loads(proc.stdout.decode('utf-8'))['text']

        raise ValueError(f'Unsupported generation backend mode: {self.mode}')

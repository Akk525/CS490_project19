import base64
import json
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.models.generation_base import GenerationBase
from src.models.response_normalizer import normalize_generation_text
from src.utils.http_retry import post_with_retries


class SchoolServerGeneration(GenerationBase):
    def __init__(self, backend_cfg: Dict, model_name: str):
        self.backend_cfg = backend_cfg
        self.model_name = model_name
        self.mode = backend_cfg['mode']

    def _build_openai_messages(self, prompt: str, image_paths: Optional[List[str]]) -> List[Dict]:
        if not image_paths:
            return [{'role': 'user', 'content': prompt}]

        content: List[Dict] = [{'type': 'text', 'text': prompt}]
        for raw_path in image_paths:
            path = str(raw_path).strip()
            if not path:
                continue
            if path.startswith('http://') or path.startswith('https://') or path.startswith('data:'):
                url = path
            else:
                file_path = Path(path).expanduser().resolve()
                if not file_path.exists():
                    raise FileNotFoundError(f'Image path does not exist: {file_path}')
                mime_type = mimetypes.guess_type(file_path.name)[0] or 'image/jpeg'
                encoded = base64.b64encode(file_path.read_bytes()).decode('ascii')
                url = f'data:{mime_type};base64,{encoded}'
            content.append({'type': 'image_url', 'image_url': {'url': url}})
        return [{'role': 'user', 'content': content}]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        image_paths: Optional[List[str]] = None,
    ) -> str:
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
                'messages': self._build_openai_messages(prompt, image_paths),
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
            resp = post_with_retries(
                url,
                headers=headers,
                json=payload,
                timeout=self.backend_cfg.get('timeout_seconds', 180),
            )
            return normalize_generation_text(resp.json())

        if self.mode == 'http':
            if image_paths:
                raise ValueError('Multimodal image input is only supported for openai_compatible generation backends.')
            cfg = self.backend_cfg['http']
            base_url = os.getenv(cfg['base_url_env'])
            url = base_url.rstrip('/') + cfg.get('endpoint', '/generate')
            headers = {'Content-Type': 'application/json'}
            auth_header = os.getenv(cfg.get('auth_header_env', ''), '')
            auth_token = os.getenv(cfg.get('auth_token_env', ''), '')
            if auth_header and auth_token:
                headers[auth_header] = auth_token
            payload = {'model': self.model_name, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature}
            resp = post_with_retries(
                url,
                headers=headers,
                json=payload,
                timeout=self.backend_cfg.get('timeout_seconds', 180),
            )
            return normalize_generation_text(resp.json())

        if self.mode == 'cli':
            if image_paths:
                raise ValueError('Multimodal image input is only supported for openai_compatible generation backends.')
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

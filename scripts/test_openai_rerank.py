import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env() -> Dict[str, str]:
    load_dotenv()
    base_url = os.getenv("SCHOOL_OPENAI_BASE_URL")
    api_key = os.getenv("SCHOOL_OPENAI_API_KEY")
    if not base_url:
        raise ValueError("Missing SCHOOL_OPENAI_BASE_URL in environment.")
    if not api_key:
        raise ValueError("Missing SCHOOL_OPENAI_API_KEY in environment.")
    return {"base_url": base_url, "api_key": api_key}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test school OpenAI-compatible rerank endpoint (/v1/rerank).")
    parser.add_argument("--model", default="hf.co/jinaai/jina-reranker-v3-GGUF:BF16")
    parser.add_argument("--query", default="A recipe query about making something tasty")
    parser.add_argument(
        "--documents",
        nargs="+",
        default=[
            "Cook pasta in boiling water, then mix with sauce.",
            "Bake a cake using flour, sugar, and eggs.",
            "Sauté vegetables in a pan and season.",
        ],
    )
    parser.add_argument("--max-docs", type=int, default=10)
    args = parser.parse_args()

    env = _load_env()
    base_url = env["base_url"].rstrip("/")
    endpoint = base_url + "/rerank"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {env['api_key']}"}
    payload: Dict[str, Any] = {"model": args.model, "query": args.query, "documents": args.documents[: args.max_docs]}

    print("Testing rerank endpoint...")
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print("Success!")
    print("Response keys:", list(data.keys()))
    # We don't know the exact schema; print truncated payload.
    print("Response (truncated):", str(data)[:1000])


if __name__ == "__main__":
    main()


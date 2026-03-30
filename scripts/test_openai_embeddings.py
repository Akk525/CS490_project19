import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test school OpenAI-compatible embeddings endpoint.")
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["Boil water and cook pasta until tender."],
        help="One or more input texts to embed.",
    )
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv("SCHOOL_OPENAI_BASE_URL")
    api_key = os.getenv("SCHOOL_OPENAI_API_KEY")
    if not base_url:
        raise ValueError("Missing SCHOOL_OPENAI_BASE_URL in environment.")
    if not api_key:
        raise ValueError("Missing SCHOOL_OPENAI_API_KEY in environment.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    print("Testing embeddings connection to school OpenAI-compatible proxy...")
    response = client.embeddings.create(model=args.model, input=args.input)
    print("Success!")
    print(f"Model Used: {response.model}")
    print(f"Embedding Count: {len(response.data)}")
    if response.data:
        print(f"First Embedding Length: {len(response.data[0].embedding)}")


if __name__ == "__main__":
    main()

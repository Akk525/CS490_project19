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
    parser = argparse.ArgumentParser(description='Test school OpenAI-compatible proxy connectivity.')
    parser.add_argument('--model', default='gemma3:4b-it-q8_0')
    parser.add_argument('--prompt', default='Hello!')
    parser.add_argument('--max-tokens', type=int, default=50)
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv('SCHOOL_OPENAI_BASE_URL')
    api_key = os.getenv('SCHOOL_OPENAI_API_KEY')
    if not base_url:
        raise ValueError('Missing SCHOOL_OPENAI_BASE_URL in environment.')
    if not api_key:
        raise ValueError('Missing SCHOOL_OPENAI_API_KEY in environment.')

    client = OpenAI(base_url=base_url, api_key=api_key)
    print('Testing connection to school OpenAI-compatible proxy...')
    response = client.chat.completions.create(
        model=args.model,
        messages=[{'role': 'user', 'content': args.prompt}],
        max_tokens=args.max_tokens,
    )
    content = response.choices[0].message.content if response.choices else ''
    print('Success!')
    print(f'Model Used: {response.model}')
    print(f'Response: {content}')


if __name__ == '__main__':
    main()

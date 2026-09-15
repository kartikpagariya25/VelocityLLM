import argparse
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="VelocityLLM Stream Viewer")
    parser.add_argument("--url", default="http://localhost:8000/generate/stream")
    parser.add_argument("--api-key", default="velocityllm-secret-123")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    payload = {"prompt": args.prompt, "max_tokens": args.max_tokens}
    headers = {"Content-Type": "application/json", "X-API-Key": args.api_key}

    print("=" * 60)
    print("  STREAMING RESPONSE")
    print("=" * 60)
    print(f"  Prompt: {args.prompt}")
    print("-" * 60)
    sys.stdout.write("  ")

    with requests.post(args.url, headers=headers, json=payload, stream=True) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: "):]
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "error" in chunk:
                print(f"\n\n  [ERROR] {chunk['error']}: {chunk.get('reason', '')}")
                return

            delta = chunk.get("delta", "")
            sys.stdout.write(delta)
            sys.stdout.flush()

            if chunk.get("is_finished"):
                break

    print()
    print("-" * 60)
    print("  Stream complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

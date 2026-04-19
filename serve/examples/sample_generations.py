#!/usr/bin/env python3
"""Sample prompts against a running vLLM OpenAI-compatible server."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from serve.client import stream_completion  # noqa: E402
from openai import OpenAI  # noqa: E402


def main() -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("SERVED_MODEL_NAME") or os.environ.get("MODEL")
    if not model:
        print("Set MODEL or SERVED_MODEL_NAME to your served checkpoint id.")
        sys.exit(2)
    client = OpenAI(base_url=base_url, api_key=api_key)

    prompts = [
        "What is 17 * 23? Answer with just the number.",
        "Name one planet in the solar system smaller than Earth.",
        "Translate to French: Good morning.",
    ]
    for i, prompt in enumerate(prompts):
        text, ttft, _ = stream_completion(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.3,
            top_p=1.0,
            stop=["\n\n"],
        )
        print(f"--- Sample {i+1} ---")
        print(f"Q: {prompt}\nA: {text.strip()}")
        if ttft:
            print(f"(TTFT {ttft*1000:.0f} ms)\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Determinism checks and lightweight output validation for the custom JSON QA task."""

from __future__ import annotations

import argparse
import os
import re
import sys

from openai import OpenAI

# Shared with eval: short answers only
ANSWER_PATTERN = re.compile(r"^[A-Za-z0-9\s\-+.']{1,80}$")


def validate_custom_answer(text: str) -> bool:
    t = text.strip().split("\n")[0].strip()
    if not t:
        return False
    return bool(ANSWER_PATTERN.match(t))


def deterministic_double_run(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    seed: int,
) -> tuple[str, str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    common = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        extra_body={"seed": seed},
    )
    r1 = client.chat.completions.create(**common).choices[0].message.content or ""
    r2 = client.chat.completions.create(**common).choices[0].message.content or ""
    return r1, r2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.environ.get("MODEL", ""))
    p.add_argument("--seed", type=int, default=424242)
    args = p.parse_args()
    if not args.model:
        print("Set MODEL or --model")
        sys.exit(2)

    ok_samples = ["Paris", "H2O", "27"]
    bad_samples = ["", "a" * 200]
    for s in ok_samples:
        assert validate_custom_answer(s), s
    for s in bad_samples:
        assert not validate_custom_answer(s), s
    print("Regex validation: OK")

    a, b = deterministic_double_run(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt="Reply with exactly: deterministic-test",
        seed=args.seed,
    )
    if a.strip() == b.strip():
        print("Determinism: PASS (identical completions)")
    else:
        print("Determinism: MISMATCH")
        print("Run1:", repr(a[:200]))
        print("Run2:", repr(b[:200]))
        sys.exit(1)


if __name__ == "__main__":
    main()

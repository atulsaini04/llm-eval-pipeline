#!/usr/bin/env python3
"""Generative HellaSwag: baseline (greedy) vs self-consistency (majority vote)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from improve.optimize_prompt import (  # noqa: E402
    baseline_prompt,
    baseline_prompt_fewshot,
    cot_prompt,
    cot_prompt_fewshot,
    load_subset,
    select_few_shot_examples,
)

DIGIT_RE = re.compile(r"[0-3]")


def parse_choice(text: str) -> int | None:
    if not text:
        return None
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    for line in reversed(lines):
        for tok in line.split():
            m = DIGIT_RE.search(tok)
            if m:
                return int(m.group(0))
    m = DIGIT_RE.search(text)
    return int(m.group(0)) if m else None


def predict_one(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    system_message: str = "",
    temperature: float,
    seed_base: int,
    sample_idx: int,
) -> int | None:
    msgs: list[dict[str, str]] = []
    if system_message.strip():
        msgs.append({"role": "system", "content": system_message.strip()})
    msgs.append({"role": "user", "content": prompt})
    r = client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=96,
        temperature=temperature,
        top_p=1.0 if temperature == 0 else 0.95,
        extra_body={"seed": seed_base + sample_idx},
    )
    text = r.choices[0].message.content or ""
    return parse_choice(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="improve/data/hellaswag_subset.jsonl")
    p.add_argument("--output", default="improve/runs/predictions.jsonl")
    p.add_argument("--model", default="")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--prompt", choices=("baseline", "cot"), default="baseline")
    p.add_argument("--mode", choices=("baseline", "vote"), default="baseline")
    p.add_argument("--k", type=int, default=5, help="Samples for majority vote.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--fewshot",
        action="store_true",
        help="Prepend deterministic few-shot examples (Jaccard-selected from pool).",
    )
    p.add_argument("--fewshot-pool", default="improve/data/fewshot_pool.jsonl")
    p.add_argument("--fewshot-k", type=int, default=2)
    p.add_argument(
        "--system-message",
        default=os.environ.get("INFER_SYSTEM_MESSAGE", ""),
        help="Optional system message (e.g. SmolLM3: `/no_think` to disable long reasoning blocks).",
    )
    args = p.parse_args()
    model = args.model or os.environ.get("MODEL", "")
    if not model:
        raise SystemExit("Set MODEL or --model")

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    fn = baseline_prompt if args.prompt == "baseline" else cot_prompt
    pool: list[dict] = []
    if args.fewshot:
        pool = load_subset(Path(args.fewshot_pool))

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            if args.fewshot and pool:
                ex = select_few_shot_examples(row["ctx"], pool, k=args.fewshot_k)
                if args.prompt == "baseline":
                    prompt = baseline_prompt_fewshot(row["ctx"], row["endings"], ex)
                else:
                    prompt = cot_prompt_fewshot(row["ctx"], row["endings"], ex)
            else:
                prompt = fn(row["ctx"], row["endings"])
            label = int(row["label"])
            if args.mode == "baseline":
                pred = predict_one(
                    client,
                    model,
                    prompt,
                    system_message=args.system_message,
                    temperature=0.0,
                    seed_base=args.seed,
                    sample_idx=row["id"],
                )
            else:
                votes: list[int] = []
                for j in range(args.k):
                    pr = predict_one(
                        client,
                        model,
                        prompt,
                        system_message=args.system_message,
                        temperature=0.6,
                        seed_base=args.seed + 1000 * row["id"],
                        sample_idx=j,
                    )
                    if pr is not None:
                        votes.append(pr)
                pred = max(set(votes), key=votes.count) if votes else None
            n += 1
            if pred == label:
                correct += 1
            out.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "label": label,
                        "pred": pred,
                        "prompt_mode": args.prompt,
                        "infer_mode": args.mode,
                        "fewshot": bool(args.fewshot),
                    }
                )
                + "\n"
            )
    acc = correct / max(n, 1)
    print(json.dumps({"accuracy": acc, "n": n, "output": str(out_path)}))


if __name__ == "__main__":
    main()

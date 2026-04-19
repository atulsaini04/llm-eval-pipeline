#!/usr/bin/env python3
"""Prompt templates and optional few-shot / CoT variants for generative HellaSwag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def baseline_prompt(ctx: str, endings: list[str]) -> str:
    lines = "\n".join(f"{i}. {e}" for i, e in enumerate(endings))
    return (
        "Which ending makes the most sense? Reply with a single digit 0, 1, 2, or 3 only.\n\n"
        f"Context: {ctx}\n\n"
        f"{lines}\n\n"
        "Answer:"
    )


def cot_prompt(ctx: str, endings: list[str]) -> str:
    lines = "\n".join(f"{i}. {e}" for i, e in enumerate(endings))
    return (
        "Read the context and four candidate endings. Think step by step which is most plausible, "
        "then on the last line output only one digit 0, 1, 2, or 3.\n\n"
        f"Context: {ctx}\n\n"
        f"{lines}\n\n"
        "Reason briefly, then last line: Answer:"
    )


def load_subset(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _words(s: str) -> set[str]:
    return {w for w in s.lower().split() if w}


def jaccard_similarity(a: str, b: str) -> float:
    A, B = _words(a), _words(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def select_few_shot_examples(ctx: str, pool: list[dict], k: int = 2) -> list[dict]:
    """Deterministic few-shot selection: rank pool by Jaccard overlap on context (no embeddings)."""
    scored = [(jaccard_similarity(ctx, ex["ctx"]), i, ex) for i, ex in enumerate(pool)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored[:k]]


def format_few_shot_block(examples: list[dict]) -> str:
    blocks = []
    for ex in examples:
        lines = "\n".join(f"{i}. {e}" for i, e in enumerate(ex["endings"]))
        blocks.append(
            f"Example:\nContext: {ex['ctx']}\n{lines}\nCorrect index: {ex['label']}\n"
        )
    return "\n".join(blocks)


def baseline_prompt_fewshot(ctx: str, endings: list[str], examples: list[dict]) -> str:
    return format_few_shot_block(examples) + "\n" + baseline_prompt(ctx, endings)


def cot_prompt_fewshot(ctx: str, endings: list[str], examples: list[dict]) -> str:
    return format_few_shot_block(examples) + "\n" + cot_prompt(ctx, endings)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="improve/data/hellaswag_subset.jsonl")
    p.add_argument("--mode", choices=("baseline", "cot"), default="baseline")
    args = p.parse_args()
    rows = load_subset(Path(args.data))
    p0 = baseline_prompt(rows[0]["ctx"], rows[0]["endings"])
    if args.mode == "cot":
        p0 = cot_prompt(rows[0]["ctx"], rows[0]["endings"])
    print("Sample prompt:\n", p0[:1200])


if __name__ == "__main__":
    main()

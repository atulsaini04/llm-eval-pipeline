#!/usr/bin/env python3
"""Export a fixed HellaSwag subset for reproducible inference-time experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="improve/data/hellaswag_subset.jsonl")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    pick = sorted(indices[: args.n])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in pick:
            row = ds[i]
            rec = {
                "id": i,
                "ctx": row["ctx"],
                "endings": row["endings"],
                "label": int(row["label"]),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(pick)} rows to {out}")


if __name__ == "__main__":
    main()

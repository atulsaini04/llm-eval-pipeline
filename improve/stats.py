#!/usr/bin/env python3
"""Paired statistics: bootstrap 95% CI for accuracies and McNemar test for difference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def load_pairs(baseline_path: Path, improved_path: Path) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Returns (baseline_correct, improved_correct) as 0/1 arrays aligned by id."""
    b_map: dict[int, int] = {}
    with open(baseline_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            b_map[int(r["id"])] = int(r["pred"] == r["label"])
    bc: list[int] = []
    ic: list[int] = []
    with open(improved_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            iid = int(r["id"])
            if iid not in b_map:
                continue
            bc.append(b_map[iid])
            ic.append(int(r["pred"] == r["label"]))
    return np.array(bc, dtype=np.int8), np.array(ic, dtype=np.int8)


def bootstrap_ci_diff(
    bc: np.ndarray, ic: np.ndarray, *, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    """95% CI for (mean(improved) - mean(baseline)) on paired items."""
    rng = np.random.default_rng(seed)
    n = len(bc)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(ic.mean() - bc.mean())
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(ic[idx].mean() - bc[idx].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, float(lo), float(hi)


def mcnemar_paired(bc: np.ndarray, ic: np.ndarray):
    """McNemar test (chi² with continuity correction) on discordant pairs."""
    if len(bc) == 0:
        return float("nan"), float("nan")
    n01 = int(np.sum((bc == 0) & (ic == 1)))
    n10 = int(np.sum((bc == 1) & (ic == 0)))
    if n01 + n10 == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    pval = float(1 - chi2.cdf(stat, df=1))
    return float(stat), pval


def wilson_ci(acc: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = acc
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(center - half), float(center + half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="improve/runs/baseline.jsonl")
    p.add_argument("--improved", default="improve/runs/improved.jsonl")
    p.add_argument("--out", default="improve/runs/stats.json")
    args = p.parse_args()
    bp = Path(args.baseline)
    ip = Path(args.improved)
    if not bp.is_file() or not ip.is_file():
        print("Run improve/eval.sh first to produce baseline and improved JSONL files.")
        raise SystemExit(1)
    bc, ic = load_pairs(bp, ip)
    n = len(bc)
    acc_b = float(bc.mean()) if n else 0.0
    acc_i = float(ic.mean()) if n else 0.0
    obs, lo, hi = bootstrap_ci_diff(bc, ic)
    mstat, pval = mcnemar_paired(bc, ic)
    ci_b = wilson_ci(acc_b, n)
    ci_i = wilson_ci(acc_i, n)
    out = {
        "n": n,
        "baseline_accuracy": acc_b,
        "improved_accuracy": acc_i,
        "delta_accuracy": acc_i - acc_b,
        "bootstrap_95ci_delta": [lo, hi],
        "wilson_95ci_baseline": list(ci_b),
        "wilson_95ci_improved": list(ci_i),
        "mcnemar_statistic": mstat,
        "mcnemar_pvalue": pval,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

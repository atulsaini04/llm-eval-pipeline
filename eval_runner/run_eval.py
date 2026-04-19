#!/usr/bin/env python3
"""Run lm-evaluation-harness against a local vLLM OpenAI Completions endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root (parent of eval_runner/)
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "eval_runner"))

import vllm_model  # noqa: F401 — registers `vllm-remote`

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402


def _write_summary(results: dict, path: Path) -> None:
    lines = ["# Evaluation summary\n", "| Task | Metric | Value |\n", "|------|--------|-------|\n"]
    r = results.get("results", {})
    for task_name, task_res in r.items():
        if not isinstance(task_res, dict):
            continue
        for k, v in task_res.items():
            if k.startswith("_") or isinstance(v, dict):
                continue
            if isinstance(v, (int, float)):
                lines.append(f"| {task_name} | {k} | {v:.4f} |\n")
            else:
                lines.append(f"| {task_name} | {k} | {v} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL", "HuggingFaceTB/SmolLM3-3B"),
        help="HuggingFace model id passed to vLLM (must match served name).",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_COMPLETIONS_URL", "http://127.0.0.1:8000/v1/completions"),
    )
    p.add_argument("--tasks", default="hellaswag,mmlu_stem,custom_json_qa")
    p.add_argument("--limit", type=float, default=None, help="Per-task example limit (dev).")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--batch-size", default="1")
    p.add_argument("--num-concurrent", type=int, default=4)
    p.add_argument("--output", default="results/eval_results.json")
    p.add_argument("--summary", default="results/summary.md")
    p.add_argument("--use-cache", default="eval_runner/cache/lm_harness.sqlite")
    p.add_argument("--no-cache-requests", action="store_true")
    p.add_argument("--disk-cache", default="eval_runner/cache/vllm_api")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument(
        "--max-length",
        type=int,
        default=int(os.environ.get("LM_EVAL_MAX_LENGTH", "2048")),
        help="Must be <= vLLM --max-model-len (Colab default 2048).",
    )
    args = p.parse_args()

    # trust_remote_code: required for SmolLM3 and many recent HF tokenizers when using huggingface backend.
    # Do not put batch_size in model_args: simple_evaluate() passes it via
    # additional_config and would duplicate the keyword (TypeError).
    model_args = ",".join(
        [
            f"model={args.model}",
            f"base_url={args.base_url}",
            f"tokenizer={args.model}",
            "tokenizer_backend=huggingface",
            f"num_concurrent={args.num_concurrent}",
            f"cache_dir={args.disk_cache}",
            f"max_length={args.max_length}",
            "trust_remote_code=true",
        ]
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tm = TaskManager(include_path=str(ROOT / "eval_runner" / "tasks"), include_defaults=True)

    try:
        results = simple_evaluate(
            model="vllm-remote",
            model_args=model_args,
            tasks=task_list,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            use_cache=args.use_cache,
            cache_requests=not args.no_cache_requests,
            task_manager=tm,
            bootstrap_iters=args.bootstrap_iters,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42,
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    if results is None:
        raise RuntimeError("simple_evaluate returned None.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    _write_summary(results, Path(args.summary))
    print(f"Wrote {out} and {args.summary}")


if __name__ == "__main__":
    main()

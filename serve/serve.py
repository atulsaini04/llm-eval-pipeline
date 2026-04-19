#!/usr/bin/env python3
"""Launch vLLM OpenAI-compatible server (PagedAttention + continuous batching are defaults)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Start vLLM server (execs `vllm serve`).")
    p.add_argument("--model", default=os.environ.get("MODEL", "HuggingFaceTB/SmolLM3-3B"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args, extra = p.parse_known_args()

    vllm = shutil.which("vllm")
    if not vllm:
        print(
            "ERROR: `vllm` not found on PATH. Install with: pip install vllm",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        vllm,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    cmd.extend(extra)
    os.execvp(vllm, cmd)


if __name__ == "__main__":
    main()

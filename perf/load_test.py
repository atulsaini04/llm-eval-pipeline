#!/usr/bin/env python3
"""Concurrent load generator: TTFT, tokens/s, latency percentiles, optional GPU util."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

try:
    import pynvml
except ImportError:
    pynvml = None


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


SHORT_PROMPTS = [
    "Say OK.",
    "Name a color.",
    "1+1=?",
    "Capital of Italy?",
    "Hi in Spanish?",
]
LONG_PREFIX = (
    "Summarize the following paragraph in one sentence:\n\n"
    "Machine learning systems often trade off latency and accuracy. "
    "Batching increases throughput but can delay individual requests. "
)


@dataclass
class Row:
    """Assignment metrics: TTFT; throughput (tokens/s); latency p50/p95/p99; optional GPU.

    Note: the brief uses “TPOT” for tokens/sec in places; we log `throughput_tokens_per_sec`
    and median time-per-output-token as `tpot_ms_per_token` (ms/token).
    `client_batch_size` mirrors `concurrency` (client-side parallelism / batch size).
    """

    run_id: str
    profile: str
    concurrency: int
    client_batch_size: int
    cache_on: bool
    stop_strategy: str
    ttft_ms_p50: float
    throughput_tokens_per_sec: float
    tpot_ms_per_token: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    gpu_util_avg: str


def _stream_chat(
    client: httpx.Client,
    *,
    base: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
) -> tuple[float | None, float, int]:
    """Returns (ttft_sec, total_sec, num_output_tokens estimated from response)."""
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if stop:
        payload["stop"] = stop
    t0 = time.perf_counter()
    ttft: float | None = None
    n_chars = 0
    with client.stream("POST", url, headers=headers, json=payload, timeout=600.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for ch in chunk.get("choices", []):
                delta = ch.get("delta") or {}
                c = delta.get("content") or ""
                n_chars += len(c)
    total = time.perf_counter() - t0
    # Rough token estimate ~4 chars per token for English
    n_tok = max(1, n_chars // 4)
    return ttft, total, n_tok


def _gpu_sampler(stop_evt: threading.Event, out: list[float]) -> None:
    if pynvml is None:
        return
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop_evt.wait(0.2):
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            out.append(float(util.gpu))
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def run_once(
    *,
    profile: str,
    concurrency: int,
    cache_on: bool,
    stop_strategy: str,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> Row:
    run_id = str(uuid.uuid4())[:8]
    if profile == "long":
        prompts = [LONG_PREFIX + f"Variant {i}: " + random.choice(SHORT_PROMPTS) for i in range(concurrency)]
    else:
        prompts = [random.choice(SHORT_PROMPTS) for _ in range(concurrency)]

    stop = ["\n\n"] if stop_strategy == "double_newline" else None
    # cache_on: repeat same prompt set to hit server prefix cache / our client-side memo
    if cache_on:
        prompts = [prompts[0]] * concurrency

    ttfts: list[float] = []
    totals: list[float] = []
    tok_s_list: list[float] = []
    tpot_ms_list: list[float] = []

    gpu_samples: list[float] = []
    stop_evt = threading.Event()
    t_gpu = threading.Thread(target=_gpu_sampler, args=(stop_evt, gpu_samples), daemon=True)
    t_gpu.start()

    with httpx.Client() as client:
        for p in prompts:
            ttft, total, n_tok = _stream_chat(
                client,
                base=base_url,
                model=model,
                api_key=api_key,
                prompt=p,
                max_tokens=max_tokens,
                temperature=0.7 if not cache_on else 0.0,
                top_p=1.0,
                stop=stop,
            )
            if ttft is not None:
                ttfts.append(ttft * 1000)
            totals.append(total * 1000)
            gen_time = max(total - (ttft or 0), 1e-6)
            tok_s_list.append(n_tok / gen_time)
            tpot_ms_list.append((gen_time * 1000) / max(n_tok, 1))

    stop_evt.set()
    t_gpu.join(timeout=1.0)

    ttfts.sort()
    totals.sort()
    ttft_p50 = _percentile(ttfts, 50) if ttfts else float("nan")
    lat_p50 = _percentile(totals, 50)
    lat_p95 = _percentile(totals, 95)
    lat_p99 = _percentile(totals, 99)
    throughput = statistics.median(tok_s_list) if tok_s_list else float("nan")
    tpot_ms = statistics.median(tpot_ms_list) if tpot_ms_list else float("nan")
    gpu_avg = f"{statistics.mean(gpu_samples):.1f}" if gpu_samples else "na"

    return Row(
        run_id=run_id,
        profile=profile,
        concurrency=concurrency,
        client_batch_size=concurrency,
        cache_on=cache_on,
        stop_strategy=stop_strategy,
        ttft_ms_p50=ttft_p50,
        throughput_tokens_per_sec=throughput,
        tpot_ms_per_token=tpot_ms,
        latency_ms_p50=lat_p50,
        latency_ms_p95=lat_p95,
        latency_ms_p99=lat_p99,
        gpu_util_avg=gpu_avg,
    )


def append_csv(path: str, row: Row) -> None:
    file_exists = os.path.isfile(path)
    fields = list(Row.__dataclass_fields__)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow({k: getattr(row, k) for k in fields})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.environ.get("MODEL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", default="metrics.csv")
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    args = p.parse_args()
    if not args.model:
        print("Set --model or MODEL")
        sys.exit(2)

    for c in args.concurrency:
        for profile in ("short", "long"):
            for cache_on in (False, True):
                for stop_strategy in ("none", "double_newline"):
                    row = run_once(
                        profile=profile,
                        concurrency=c,
                        cache_on=cache_on,
                        stop_strategy=stop_strategy,
                        base_url=args.base_url,
                        model=args.model,
                        api_key=args.api_key,
                        max_tokens=args.max_tokens,
                    )
                    append_csv(args.output, row)
                    print(row)


if __name__ == "__main__":
    main()

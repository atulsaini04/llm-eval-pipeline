"""OpenAI-compatible client for vLLM: streaming, decoding params, TTFT timing."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def stream_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stop: list[str] | None = None,
) -> tuple[str, float | None, int]:
    """Returns (full_text, ttft_seconds, num_chunks)."""
    t0 = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    n_chunks = 0
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop or [],
        stream=True,
    )
    for chunk in stream:
        n_chunks += 1
        if ttft is None:
            ttft = time.perf_counter() - t0
        delta = chunk.choices[0].delta
        if delta and delta.content:
            parts.append(delta.content)
    return "".join(parts), ttft, n_chunks


def _p50(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[len(s) // 2]


def concurrent_smoke(
    client: OpenAI,
    model: str,
    *,
    n: int = 8,
    max_tokens: int = 64,
) -> None:
    prompts = [f"Count from 1 to 5, request #{i}. Reply in one line." for i in range(n)]

    def one(prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        dt = time.perf_counter() - t0
        return r.choices[0].message.content or "", dt

    seq_lat: list[float] = []
    for p in prompts:
        _, dt = one(p)
        seq_lat.append(dt)

    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(one, p) for p in prompts]
        conc_lat: list[float] = []
        for f in as_completed(futs):
            _, dt = f.result()
            conc_lat.append(dt)

    p50_seq = _p50(seq_lat)
    p50_conc = _p50(conc_lat)
    wall_seq = sum(seq_lat)
    wall_conc = max(conc_lat) if conc_lat else 0.0
    ratio = p50_conc / p50_seq if p50_seq > 1e-9 else float("nan")
    print(
        f"Sequential (one client): sum wall {wall_seq:.3f}s, per-request p50 {p50_seq:.3f}s"
    )
    print(
        f"Concurrent ({n} workers): wall (slowest) {wall_conc:.3f}s, per-request p50 {p50_conc:.3f}s"
    )
    print(
        f"Ratio concurrent_p50 / sequential_p50 = {ratio:.2f}x "
        "(healthy servers often stay near 1–2×; large spikes may indicate saturation)."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", os.environ.get("MODEL", "")))
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--stop", nargs="*", default=[])
    p.add_argument(
        "--concurrent-test",
        action="store_true",
        help="Run N distinct prompts sequentially then concurrently; compare p50 latencies.",
    )
    args = p.parse_args()

    if not args.model:
        print("Set --model or MODEL / SERVED_MODEL_NAME to the HuggingFace id served by vLLM.")
        raise SystemExit(2)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.concurrent_test:
        concurrent_smoke(client, args.model)
        return

    text, ttft, chunks = stream_completion(
        client,
        model=args.model,
        messages=[{"role": "user", "content": "Say hello in exactly five words."}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=list(args.stop) or None,
    )
    print(text)
    if ttft is not None:
        print(f"TTFT: {ttft*1000:.1f} ms, stream chunks: {chunks}")


if __name__ == "__main__":
    main()

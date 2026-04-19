# Part A — vLLM serving

vLLM exposes an **OpenAI-compatible HTTP API** (not a legacy `/generate` path). Generation is available at:

| Method | Path | Role |
|--------|------|------|
| `POST` | `/v1/completions` | Text prompts; used by lm-eval **local-completions** / `vllm-remote` for log-likelihood tasks |
| `POST` | `/v1/chat/completions` | Chat messages; used by [`client.py`](client.py) streaming demos |

PagedAttention and continuous batching are **enabled by default** in vLLM; no extra flags are required for those features.

**Startup:** from repo root, `make serve` or `python serve/serve.py --model "$MODEL" --host 0.0.0.0 --port 8000`.

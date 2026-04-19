# Guardrails & determinism

## What we test

1. **Deterministic decoding** (`validate.py`): Two back-to-back chat completions with the same `temperature=0`, `top_p=1`, and fixed `seed`. The script expects byte-identical text; if the server ignores `seed` or uses nondeterministic kernels, the check fails and prints both outputs.

2. **Custom benchmark output shape**: A regex allows short factual answers (letters, numbers, basic punctuation) and rejects empty or pathologically long answers.

## Where nondeterminism still appears

- **FlashAttention / fused kernels** on some GPUs may still vary slightly without full deterministic mode in the inference stack.
- **Concurrent scheduling** under load can change batching order; this script uses sequential requests.
- **Floating-point** differences across driver versions are rare for greedy decoding but possible in sampling.
- If vLLM is started without features that enforce bitwise reproducibility, treat `seed` as best-effort.

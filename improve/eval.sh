#!/usr/bin/env bash
# Run baseline vs improved (self-consistency) on the same HellaSwag subset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

MODEL="${MODEL:-HuggingFaceTB/SmolLM3-3B}"
export MODEL

# SmolLM3 uses a long "thinking" chat template; disable for short digit answers (see HF model card).
SYS_ARGS=()
if [[ "$MODEL" == *"SmolLM3"* ]]; then
  SYS_ARGS=(--system-message "${INFER_SYSTEM_MESSAGE:-/no_think}")
fi

echo "=== Preparing subset (HellaSwag — best lever for inference-only gains on this repo) ==="
"$PY" improve/prepare_data.py --n "${SUBSET_N:-100}" --output improve/data/hellaswag_subset.jsonl

echo "=== Baseline (greedy, baseline prompt) ==="
"$PY" improve/infer.py \
  --model "$MODEL" \
  "${SYS_ARGS[@]}" \
  --prompt baseline \
  --mode baseline \
  --data improve/data/hellaswag_subset.jsonl \
  --output improve/runs/baseline.jsonl

echo "=== Improved (CoT prompt + majority vote) ==="
"$PY" improve/infer.py \
  --model "$MODEL" \
  "${SYS_ARGS[@]}" \
  --prompt cot \
  --mode vote \
  --k "${VOTE_K:-5}" \
  --data improve/data/hellaswag_subset.jsonl \
  --output improve/runs/improved.jsonl

echo "=== Aggregate + McNemar / bootstrap CI (improve/stats.py) ==="
"$PY" improve/stats.py \
  --baseline improve/runs/baseline.jsonl \
  --improved improve/runs/improved.jsonl \
  --out improve/runs/stats.json

if [[ "${RUN_HARNESS_HELLASWAG:-0}" == "1" ]]; then
  echo "=== Optional: lm-eval HellaSwag (log-likelihood) small slice — same vLLM server ==="
  "$PY" eval_runner/run_eval.py \
    --tasks hellaswag \
    --limit "${HARNESS_LIMIT:-0.01}" \
    --output improve/runs/harness_hellaswag_eval.json \
    --summary improve/runs/harness_hellaswag_summary.md
fi

# LLM evaluation pipeline (vLLM + lm-evaluation-harness)

Internal-style assignment covering **serving**, **lm-eval integration**, **load testing**, **guardrails**, and **inference-time benchmark improvement**.

## Google Colab (recommended without a local NVIDIA GPU)

Use this path if you do not have a local NVIDIA GPU (e.g. Apple Silicon). One end-to-end run typically takes **~20–60+ minutes** on a Colab **T4**, mostly install + vLLM startup + eval.

### Steps

1. **Open Colab**  
   Go to [colab.research.google.com](https://colab.research.google.com/) and sign in.

2. **Open the notebook**  
   - **File → Upload notebook** and select [`colab/LLM_Eval_Pipeline_Colab.ipynb`](colab/LLM_Eval_Pipeline_Colab.ipynb) from your clone of this repo, **or**  
   - In Colab: **File → Open notebook → GitHub** and paste your fork URL (e.g. `https://github.com/<you>/llm-eval-pipeline/blob/main/colab/LLM_Eval_Pipeline_Colab.ipynb`) so you always run the version from GitHub.

3. **Enable a GPU**  
   **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is enough; larger GPUs are faster).

4. **Configure the first code cell** (optional; defaults work for a smoke run):

   | Variable | Purpose |
   |----------|---------|
   | `GITHUB_REPO_URL` | Default clones this project from GitHub. Set to `None` if you prefer to **upload a zip** of the repo when the setup cell asks (zip must contain `serve/serve.py`, `eval_runner/`, etc.). |
   | `MODEL` | HuggingFace id served by vLLM (default **`HuggingFaceTB/SmolLM3-3B`**). |
   | `EVAL_LIMIT` | Fraction of each benchmark (default **`0.02`** keeps Part B fast). |
   | `EVAL_TASKS` | Comma-separated lm-eval tasks (default includes HellaSwag, MMLU STEM subgroup, custom JSON QA). |
   | `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_NUM_SEQS` | T4 VRAM tuning; lower further if vLLM **CUDA OOM** during warmup. |
   | `LOAD_TEST_*` | Part C load test (concurrency, `--quick` matrix). |

   **Private GitHub repo:** add a Colab secret **`GITHUB_TOKEN`** (read access to the repo) so `git pull` and the Part C self-heal can fetch updates. Alternatively use a **public** repo or **`GITHUB_REPO_URL = None`** + zip upload.

5. **Run everything**  
   **Runtime → Run all**. The first execution installs `pip` dependencies, **`vllm`**, and **`lm-eval`** (can take several minutes). Then the notebook:

   | Section | What it does |
   |---------|----------------|
   | Setup | Clones or unpacks the repo under `/content/assign`, sets env vars. |
   | Install | `pip install -r requirements.txt` + `vllm`. |
   | Part A | Starts **`vllm serve`** in the background (OpenAI-compatible API). |
   | Part B | **`eval_runner/run_eval.py`** → `results/eval_results.json`, `results/summary.md`. |
   | Part C | **`perf/load_test.py`** → **`metrics.csv`** (root of repo). |
   | Part D | **`guardrails/validate.py`**. |
   | Part E | **`improve/eval.sh`** → **`improve/runs/stats.json`** and run artifacts. |
   | Download | Builds **`assign_results.zip`** and starts a browser download. |

6. **Get your artifacts**  
   When the **Download results zip** cell finishes, your browser saves **`assign_results.zip`**. Unzip it to any folder (for example a directory named `assign_results` next to your Downloads). That folder is a snapshot of the project tree **including generated outputs**, for example:

   - **`results/eval_results.json`**, **`results/summary.md`** — lm-eval outputs  
   - **`metrics.csv`** — load-test metrics (TTFT, throughput, latency percentiles, …)  
   - **`improve/runs/stats.json`**, **`improve/runs/baseline.jsonl`**, **`improve/runs/improved.jsonl`** — Part E stats and generations  
   - **`eval_runner/cache/`** (optional, can be large) — API response cache from eval  

   Open **`perf/analysis.ipynb`** locally on the unzipped **`metrics.csv`** if you want plots.

### Troubleshooting (Colab)

OOM on vLLM, failed Step B, stale `perf/load_test.py`, disconnects: see **[`colab/README.md`](colab/README.md)**.

## Prerequisites (local)

- Python 3.10+ and an **NVIDIA GPU with CUDA** for vLLM (typical laptop dGPU is fine with a **3B** model and `--dtype auto`). **Apple Silicon:** vLLM is not fully supported the same way—use Colab above, a Linux/Windows box with NVIDIA, or a small cloud GPU for full runs.
- Recommended: create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install vllm            # install separately; large wheel
```

Set the **served model id** to match the HuggingFace name passed to `vllm serve` (used by clients and lm-eval). **Defaults are tuned for a laptop GPU** (single consumer card, limited VRAM):

| Default | Why |
|---------|-----|
| **`HuggingFaceTB/SmolLM3-3B`** | [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) (2025): 3B instruct, Apache 2.0, strong quality/size ratio, ~3B BF16 weights — fits typical notebooks with `--dtype auto` / half precision. |
| **Part E benchmark: HellaSwag (generative)** | Same underlying items as the harness task, but **digit + CoT + voting** gives the **best realistic chance** at inference-only accuracy gains vs **MMLU** (mostly log-likelihood in harness) or **ARC** (harder, smaller margins). |

**Alternative if you have more VRAM (~10GB+):** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (2025 Qwen3 line). Override: `export MODEL=Qwen/Qwen3-4B-Instruct-2507`.

```bash
export MODEL=HuggingFaceTB/SmolLM3-3B
```

For **SmolLM3**, `improve/eval.sh` passes `--system-message /no_think` so the model’s chat template does not spend tokens on long “thinking” blocks when we only need a single digit (see the model card). Override with `INFER_SYSTEM_MESSAGE` if needed.

## Part A — Serving

**One-line start** (from repo root):

```bash
make serve
# or
python serve/serve.py --model "$MODEL" --host 0.0.0.0 --port 8000
```

This execs `vllm serve` with OpenAI-compatible HTTP APIs. Continuous batching and paged attention are **on by default** in vLLM. REST paths are **`POST /v1/completions`** and **`POST /v1/chat/completions`** (there is no separate `/generate` path; these are the supported generation endpoints). See [`serve/README.md`](serve/README.md).

**Python client** (streaming + decoding parameters):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export SERVED_MODEL_NAME="$MODEL"
python serve/client.py --concurrent-test
python serve/examples/sample_generations.py
```

`--concurrent-test` runs the **same** prompts **sequentially** then **concurrently** and prints p50 latencies plus a ratio (used to sanity-check that concurrency does not explode tail latency on your GPU).

## Part B — Evaluation

With vLLM running, point lm-eval at the **Completions** endpoint (required for log-likelihood tasks such as HellaSwag and MMLU):

```bash
export VLLM_COMPLETIONS_URL=http://127.0.0.1:8000/v1/completions
python eval_runner/run_eval.py \
  --model "$MODEL" \
  --base-url "$VLLM_COMPLETIONS_URL" \
  --tasks hellaswag,mmlu_stem,custom_json_qa \
  --limit 0.01
```

Add **`arc_challenge`** to `--tasks` if you want ARC-Challenge alongside the two official tasks (MMLU subgroup + HellaSwag).

- **`vllm-remote`** in [`eval_runner/vllm_model.py`](eval_runner/vllm_model.py) subclasses the harness `local-completions` stack and adds a **disk cache** of raw JSON responses (keyed by request payload hash).
- **Custom JSON benchmark**: [`eval_runner/tasks/custom_json_qa/custom_json_qa.yaml`](eval_runner/tasks/custom_json_qa/custom_json_qa.yaml) + [`eval_runner/data/custom_benchmark.jsonl`](eval_runner/data/custom_benchmark.jsonl).
- Outputs: [`results/eval_results.json`](results/eval_results.json), [`results/summary.md`](results/summary.md).

Full benchmarks are slow; use `--limit` for development (float = fraction per task).

## Part C — Performance

```bash
export MODEL=...
python perf/load_test.py --output metrics.csv --model "$MODEL" --base-url http://127.0.0.1:8000/v1 --completions
```

Use **`--completions`** to hit `/v1/completions` (same API family as lm-eval / Part B). Without it, the script uses `/v1/chat/completions`. **`--quick`** runs a smaller matrix (good for Colab); **`--overwrite`** replaces `metrics.csv` instead of appending.

Open [`perf/analysis.ipynb`](perf/analysis.ipynb) to plot `metrics.csv`. Columns include TTFT (p50), **`throughput_tokens_per_sec`** (tokens/s), **`tpot_ms_per_token`** (median ms per output token), **`client_batch_size`** (same as concurrency), latency p50/p95/p99, and optional GPU utilization (via `pynvml` when available).

## Part D — Guardrails

```bash
python guardrails/validate.py --model "$MODEL"
```

See [`guardrails/README.md`](guardrails/README.md) for scope and known nondeterminism.

## Part E — Inference-time improvement (HellaSwag-style)

Generative multiple-choice on a **fixed subset** (exported from HellaSwag validation):

```bash
bash improve/eval.sh
```

This writes **`improve/runs/stats.json`** (bootstrap 95% CI on the paired accuracy gap, Wilson CIs, **McNemar p-value** via `improve/stats.py`). The narrative is [`improve/report.md`](improve/report.md) (400–700 words). **Assignment target lifts** (+3% HellaSwag, +2% MMLU subgroup, +2.5% ARC) are **goals**, not guarantees—report actual CIs and *p*-values.

**Optional:** `infer.py --fewshot` prepends **deterministic Jaccard-selected** examples from [`improve/data/fewshot_pool.jsonl`](improve/data/fewshot_pool.jsonl). Set `RUN_HARNESS_HELLASWAG=1` when running `eval.sh` to also snapshot **lm-eval HellaSwag** (log-likelihood) at `HARNESS_LIMIT` for comparison to the harness metric.

## Makefile targets

| Target        | Purpose                    |
|---------------|----------------------------|
| `make serve`  | Start `serve/serve.py`     |
| `make eval`   | Run `eval_runner/run_eval.py` |
| `make perf`   | Run `perf/load_test.py` → `metrics.csv` |
| `make guardrails` | Run `guardrails/validate.py` |
| `make improve`| Run `improve/eval.sh` (runs `improve/stats.py` at the end) |

## Submission checklist (Mercor)

- [ ] All folders: `serve/`, `eval_runner/`, `perf/`, `guardrails/`, `improve/`
- [ ] `Makefile`, this `README.md`, `requirements.txt`
- [ ] `results/` with eval outputs and `metrics.csv`
- [ ] Email zip or GitHub link to `mle-interviewers@mercor.com`

## Final summary (story)

This repo wires **production-style vLLM serving** to **lm-evaluation-harness** through a **cached OpenAI Completions client**, adds **load instrumentation** (TTFT, percentiles, optional GPU stats), **determinism and regex guardrails**, and a **self-consistent generative** variant for HellaSwag-style items to study accuracy–latency trade-offs without weight updates. The main learning is that **API-level caching** and **harness request caching** stack: repeated evaluation becomes cheaper and more reproducible, while **majority-vote decoding** can improve robustness at predictable multipliers of cost.

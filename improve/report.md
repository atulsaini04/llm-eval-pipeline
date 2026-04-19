# Inference-time improvement report: generative HellaSwag subset

## 1. Motivation and scope

Harness **HellaSwag** uses **log-likelihood** over endings; our **inference-only** track uses a **generative** protocol on the **same items**: output one digit `0–3`, exact match to the label. That enables CoT, voting, and few-shot levers—the **highest-leverage** choice among HellaSwag / MMLU / ARC for **prompting + decoding** tricks without weight updates (MMLU in harness is dominated by log-prob scoring; ARC is harsher). Items come from a **fixed subset** (`prepare_data.py`; default **`N=100`** in `eval.sh` for laptop runs).

**Default model:** `HuggingFaceTB/SmolLM3-3B` (mid‑2025 3B instruct, laptop-friendly). Use **`/no_think`** in the system slot for SmolLM3 so short digit answers are not buried in long reasoning blocks.

## 2. Baseline configuration

**Baseline:** `baseline_prompt` + greedy decoding (`temperature=0`, `top_p=1`, per-item `seed` via `extra_body`). **Parsing:** last-line-first digit in `0–3` (`infer.py`); missing digits count as wrong.

## 3. Improved configuration

**Improved:** `cot_prompt` (short rationale) + **k=5** samples (`temperature=0.6`, `top_p=0.95`) + **majority vote**. Optional **`--fewshot`** prepends Jaccard-selected examples from `fewshot_pool.jsonl`. Seeds: `12345 + 1000 * example_id + sample_idx`.

## 4. Results

After `bash improve/eval.sh`, read **`improve/runs/stats.json`** (from `improve/stats.py`): it reports Wilson 95% CIs per method, bootstrap 95% CI on the **paired** accuracy difference, and **McNemar’s p-value** for discordant pairs. Replace the placeholder row below with those numbers.

| Method | Accuracy | Wilson 95% CI | Δ vs baseline | Bootstrap 95% CI on Δ | McNemar p |
|--------|----------|---------------|---------------|------------------------|-----------|
| Baseline | **0.00** | [0,0] | — | — | — |
| Improved | **0.00** | [0,0] | **0.00** | [0,0] | **1.00** |

Declare significance only if **p < 0.05** (two-sided McNemar is standard for paired binary outcomes).

## 5. Ablation study

Run `infer.py` with these flags to isolate effects:

1. **Prompt only:** `--prompt cot --mode baseline` (greedy, CoT template).
2. **Voting only:** `--prompt baseline --mode vote --k 5` (same prompt as baseline, sampling + vote).
3. **Full:** `--prompt cot --mode vote` (as in `eval.sh`).

Compare deltas in accuracy and record **mean parsed digits per item** to detect parser collapse (e.g., many `None` predictions).

## 6. Before/after examples (illustrative pattern — replace IDs with your `improve/runs/*.jsonl`)

The table shows the **generative** protocol (digit 0–3). After your run, substitute real `id` values and model outputs.

| id | Gold | Baseline pred | Improved pred | Short analysis |
|----|------|---------------|---------------|----------------|
| A1 | 2 | 1 | 2 | Vote recovered correct ending after greedy picked adjacent distractor. |
| A2 | 0 | 0 | 0 | Agreement; trivial physical context. |
| A3 | 3 | 3 | 1 | Sampling broke tie; CoT sometimes overfits wrong rationale. |
| A4 | 1 | None | 1 | Parser failed on baseline (no digit); improved sample emitted “1”. |
| A5 | 2 | 2 | 2 | Agreement; vote did not hurt. |
| A6 | 0 | 2 | 0 | Majority vote corrected greedy error on ambiguous verb phrase. |
| A7 | 3 | 3 | 3 | Agreement on clear commonsense. |
| A8 | 1 | 1 | 2 | Regression: vote pushed wrong class on noisy sample. |
| A9 | 2 | 2 | 2 | Agreement. |
| A10 | 0 | 1 | 0 | Vote fixed label; baseline likely followed spurious continuation. |

Replace placeholder IDs with real `id` fields from your JSONL; add rows if needed (**10+** comparisons).

## 7. Cost and latency

Baseline: **1×** chat call per item. Improved: **~k×** calls (vote). Wall time may be **< k×** if the server batches concurrent requests; GPU cost still scales with total tokens.

## 8. Seeds and vLLM parity

`prepare_data.py --seed 12345`. Greedy uses `extra_body={"seed": ...}` per item; vote uses `12345 + 1000 * id + j`. Keep **identical** `vllm serve` flags across runs.

## 9. What we learned

Fill after runs: significance (p < 0.05), parser failures, and whether **optional few-shot** (`infer.py --fewshot`) helped.

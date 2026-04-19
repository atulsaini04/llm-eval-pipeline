# Google Colab — troubleshooting

Follow the **Google Colab** section in the [root `README.md`](../README.md) for the full step-by-step (notebook upload, GPU, config, **`assign_results.zip`** download).

The bullets below are **fixes and tuning** when something goes wrong.

## OOM / CUDA OOM (vLLM warmup)

The notebook defaults to **`--max-model-len 2048`**, **`--gpu-memory-utilization 0.78`**, and **`--max-num-seqs 32`**. Tune in the **first config cell**: `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_NUM_SEQS`. If `vllm serve` errors on `--max-num-seqs`, remove that flag from the Part A cell or check `vllm serve --help` for your vLLM version.

## Long runs / disconnects

Use Colab Pro or reduce **`EVAL_LIMIT`**, **`SUBSET_N`**, and **`EVAL_TASKS`** (e.g. drop `mmlu_stem`).

## Step B (`run_eval`) fails

Set **`EVAL_TASKS = "hellaswag,custom_json_qa"`** to skip `mmlu_stem` if a subtask or download breaks. The Part B cell prints **stdout/stderr** from `run_eval`.

## Part C (`load_test`)

Uses **`--completions`**, **`--quick`**, **`--overwrite`**; tune **`LOAD_TEST_CONCURRENCY`** / **`LOAD_TEST_QUICK`**.

## Stale `perf/load_test.py` / unknown CLI flags

Colab may keep an old `/content/assign`. The setup cell runs **`git pull --ff-only`**. Part C syncs `perf/load_test.py` via **git checkout**, **public raw URL** (`LOAD_TEST_PY_RAW_URL`), or the **GitHub Contents API** if **`GITHUB_TOKEN`** is set (Colab secret). For a **private** repo: add the token, use **`FORCE_REFRESH_COLAB_REPO = True`** once, zip upload, or make the repo public.

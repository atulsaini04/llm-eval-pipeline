# Google Colab

1. Upload **`LLM_Eval_Pipeline_Colab.ipynb`** to [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**).
2. **Runtime → Change runtime type → GPU** (T4 or better).
3. By default the notebook clones **[llm-eval-pipeline](https://github.com/atulsaini04/llm-eval-pipeline)** (`GITHUB_REPO_URL` is preset). To upload a zip instead, set `GITHUB_REPO_URL = None` in the first config cell and upload when prompted (zip must unpack to a tree containing **`serve/serve.py`**).
4. **Runtime → Run all**.

The notebook installs `vLLM` + `lm-eval`, starts `vllm serve` in the background, then runs **Part B** (eval), **Part C** (load test / `metrics.csv`), **Part D** (guardrails), **Part E** (`improve/eval.sh` + `stats.json`), and zips the repo for download.

**OOM / CUDA OOM (common on Colab T4):** The notebook now defaults to **`--max-model-len 2048`**, **`--gpu-memory-utilization 0.78`**, and **`--max-num-seqs 32`** so vLLM does not OOM during sampler warmup. Tune in the **first config cell**: `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_NUM_SEQS`. If your `vllm` build errors on `--max-num-seqs`, remove that pair from the Part A cell or run `vllm serve --help | grep max` to see the exact flag name for your version.

**Disconnects:** Long runs may need Colab Pro or fewer tasks (`EVAL_LIMIT`, `SUBSET_N`).

**Step B (`run_eval`) fails:** Re-upload the repo after pulling latest `eval_runner/run_eval.py` (adds `trust_remote_code=true` + `max_length` for SmolLM3, traceback on error). In the notebook config cell, set `EVAL_TASKS = "hellaswag,custom_json_qa"` to skip `mmlu_stem` if downloads or a subtask break. The Part B cell now prints **stdout/stderr** from `run_eval` so the real error is visible.

**Part C (`load_test`):** Uses **`--completions`** (same `/v1/completions` path as Part B), **`--quick`**, **`--overwrite`**, and prints stderr on failure. Tune `LOAD_TEST_CONCURRENCY` / `LOAD_TEST_QUICK` in the config cell.

**Stale repo / `unrecognized arguments: --completions`:** Colab may keep an old `/content/assign` copy. The setup cell runs **`git pull --ff-only`**. **Part C** checks `perf/load_test.py` and, if it lacks `--completions`, tries in order: **`git checkout origin/main -- perf/load_test.py`**, **public raw URL** (`LOAD_TEST_PY_RAW_URL`), then the **GitHub Contents API** if you add a Colab secret **`GITHUB_TOKEN`** (classic PAT with `repo` read, or fine-grained read on that repo). For a **private** repo without a token, use **`FORCE_REFRESH_COLAB_REPO = True`**, zip upload, or make the repo public so the raw URL works.

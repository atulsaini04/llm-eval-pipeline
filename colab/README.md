# Google Colab

1. Upload **`LLM_Eval_Pipeline_Colab.ipynb`** to [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**).
2. **Runtime → Change runtime type → GPU** (T4 or better).
3. Zip your local project folder so the zip unpacks to a tree that contains **`serve/serve.py`** at `…/serve/serve.py` (upload when prompted), **or** set `GITHUB_REPO_URL` in the first config cell to `git clone` your repo.
4. **Runtime → Run all**.

The notebook installs `vLLM` + `lm-eval`, starts `vllm serve` in the background, then runs **Part B** (eval), **Part C** (load test / `metrics.csv`), **Part D** (guardrails), **Part E** (`improve/eval.sh` + `stats.json`), and zips the repo for download.

**OOM / CUDA OOM (common on Colab T4):** The notebook now defaults to **`--max-model-len 2048`**, **`--gpu-memory-utilization 0.78`**, and **`--max-num-seqs 32`** so vLLM does not OOM during sampler warmup. Tune in the **first config cell**: `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_NUM_SEQS`. If your `vllm` build errors on `--max-num-seqs`, remove that pair from the Part A cell or run `vllm serve --help | grep max` to see the exact flag name for your version.

**Disconnects:** Long runs may need Colab Pro or fewer tasks (`EVAL_LIMIT`, `SUBSET_N`).

**Step B (`run_eval`) fails:** Re-upload the repo after pulling latest `eval_runner/run_eval.py` (adds `trust_remote_code=true` + `max_length` for SmolLM3, traceback on error). In the notebook config cell, set `EVAL_TASKS = "hellaswag,custom_json_qa"` to skip `mmlu_stem` if downloads or a subtask break. The Part B cell now prints **stdout/stderr** from `run_eval` so the real error is visible.

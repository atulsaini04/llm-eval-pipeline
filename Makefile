PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3; fi)
PIP ?= $(shell if [ -x .venv/bin/pip ]; then echo .venv/bin/pip; else command -v pip3; fi)
# Default: HuggingFaceTB/SmolLM3-3B (2025 small instruct, ~3B params — laptop-friendly). Override if needed.
export MODEL ?= HuggingFaceTB/SmolLM3-3B
export VLLM_HOST ?= 127.0.0.1
export VLLM_PORT ?= 8000

.PHONY: serve eval perf guardrails improve clean venv

venv:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

serve:
	$(PYTHON) serve/serve.py --model $(MODEL) --host 0.0.0.0 --port $(VLLM_PORT)

eval:
	$(PYTHON) eval_runner/run_eval.py

perf:
	$(PYTHON) perf/load_test.py --output metrics.csv

guardrails:
	$(PYTHON) guardrails/validate.py

improve:
	bash improve/eval.sh

clean:
	rm -rf eval_runner/cache .pytest_cache

"""Custom lm-eval model: vLLM OpenAI Completions API + on-disk response cache."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from aiohttp import ClientSession
from tenacity import RetryError

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI

eval_logger = logging.getLogger(__name__)


def _payload_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class DiskResponseCache:
    """Stores raw OpenAI-style JSON responses keyed by request payload."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, payload: dict) -> dict | None:
        p = self._path(_payload_hash(payload))
        if not p.is_file():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, payload: dict, response: dict) -> None:
        p = self._path(_payload_hash(payload))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(response, f)
        tmp.replace(p)


@register_model("vllm-remote")
class VLLMRemoteCachedLM(LocalCompletionsAPI):
    """Same as local-completions, but caches identical API payloads to disk."""

    def __init__(
        self,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        cache_dir = cache_dir or kwargs.pop("cache_dir", None)
        super().__init__(**kwargs)
        root = cache_dir or os.environ.get(
            "LM_VLLM_CACHE", os.path.join(os.getcwd(), "eval_runner", "cache", "vllm_api")
        )
        self._disk = DiskResponseCache(root)

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], Any],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Optional[dict]:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            cached = self._disk.get(payload)
            if cached is not None:
                return cached
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
                verify=self.verify_certificate,
            )
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            out = response.json()
            self._disk.put(payload, out)
            return out
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    async def amodel_call(
        self,
        session: ClientSession,
        sem: Any,
        messages: Any,
        *,
        generate: bool = True,
        cache_keys: list | None = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ):
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cached = self._disk.get(payload)
        outputs: dict | None = None
        acquired = await sem.acquire()
        try:
            if cached is not None:
                outputs = cached
            else:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=self.header,
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        eval_logger.warning(
                            f"API request failed! Status code: {response.status}, "
                            f"Response text: {error_text}. Retrying..."
                        )
                    response.raise_for_status()
                    outputs = await response.json()
                if outputs is not None:
                    self._disk.put(payload, outputs)

            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            cache_method = "generate_until" if generate else "loglikelihood"
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except BaseException as e:
            eval_logger.error(f"Exception:{repr(e)}, outputs={outputs}, retrying.")
            raise e
        finally:
            if acquired:
                sem.release()

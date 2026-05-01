"""Adaptive LLM client: supports Anthropic, OpenAI-compatible APIs (OpenRouter, vLLM, Together), or both.

Usage:
    # Single client (OpenRouter for everything)
    client = LLMClient.from_openai_compatible(
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Single client (local vLLM)
    client = LLMClient.from_openai_compatible(
        model="Qwen/Qwen3-Coder-Next-FP8",
        base_url="http://localhost:8235/v1",
        api_key="dummy",
    )

    # Single client (Anthropic)
    client = LLMClient.from_anthropic(model="claude-sonnet-4-20250514")

    # Multi-client (cheap model for generation, strong model for refactoring)
    clients = MultiClient(
        generation=LLMClient.from_openai_compatible(...),  # Qwen3-Coder-Next
        refactoring=LLMClient.from_anthropic(...),         # Claude Sonnet
    )
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified async LLM client. Wraps either Anthropic or OpenAI-compatible APIs."""

    def __init__(self, backend: str, client, model: str, concurrency: int = 10):
        """Don't call directly — use from_anthropic() or from_openai_compatible()."""
        self.backend = backend  # "anthropic" or "openai"
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)

    @classmethod
    def from_anthropic(cls, model: str = "claude-sonnet-4-20250514", concurrency: int = 10):
        from anthropic import AsyncAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            key_path = Path.home() / ".anthropic-usc-key.txt"
            if key_path.exists():
                api_key = key_path.read_text().strip()
        client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        return cls(backend="anthropic", client=client, model=model, concurrency=concurrency)

    @classmethod
    def from_openai_compatible(
        cls,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        concurrency: int = 30,
    ):
        from openai import AsyncOpenAI
        if not api_key:
            # Check env vars, then key file
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                key_path = Path.home() / ".openrouter-api-key.txt"
                if key_path.exists():
                    api_key = key_path.read_text().strip()
            api_key = api_key or "dummy"
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return cls(backend="openai", client=client, model=model, concurrency=concurrency)

    @classmethod
    def from_config(cls, model: str, concurrency: int = 10):
        """Auto-detect backend from model name / env vars."""
        if model.startswith("claude-"):
            return cls.from_anthropic(model=model, concurrency=concurrency)
        elif "localhost" in os.environ.get("VLLM_BASE_URL", ""):
            return cls.from_openai_compatible(
                model=model,
                base_url=os.environ["VLLM_BASE_URL"],
                concurrency=concurrency,
            )
        else:
            # Default to OpenRouter
            return cls.from_openai_compatible(model=model, concurrency=concurrency)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system: str = "",
    ) -> str:
        async with self.semaphore:
            for attempt in range(3):
                try:
                    if self.backend == "anthropic":
                        resp = await self.client.messages.create(
                            model=self.model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system if system else [],
                            messages=[{"role": "user", "content": prompt}],
                        )
                        return resp.content[0].text
                    else:  # openai-compatible
                        messages = []
                        if system:
                            messages.append({"role": "system", "content": system})
                        messages.append({"role": "user", "content": prompt})
                        resp = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        return resp.choices[0].message.content

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = any(
                        tag in err_str
                        for tag in ("rate", "429", "quota", "too many", "ratelimit")
                    )
                    if is_rate_limit:
                        wait = min(30.0, 5.0 * (2 ** attempt))
                        logger.warning("Rate limited, waiting %.1fs (attempt %d)", wait, attempt + 1)
                        await asyncio.sleep(wait)
                    elif attempt == 2:
                        raise
                    else:
                        logger.warning("API error: %s, retrying (attempt %d)", e, attempt + 1)
                        await asyncio.sleep(2.0)
        raise RuntimeError("All retries exhausted")

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system: str = "",
        cache_path: Optional[Path] = None,
        cache_key_fn: Optional[Callable[[int], str]] = None,
    ) -> List[str]:
        """Batch generation with JSONL cache for resumability."""
        # Load existing cache
        cached = {}
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    cached[obj["key"]] = obj["response"]
            logger.info("Loaded %d cached responses from %s", len(cached), cache_path)

        results = [None] * len(prompts)
        tasks = []

        for i, prompt in enumerate(prompts):
            key = cache_key_fn(i) if cache_key_fn else str(i)
            if key in cached:
                results[i] = cached[key]
                continue
            tasks.append((i, key, prompt))

        if not tasks:
            return results

        logger.info("Generating %d responses (%d cached)", len(tasks), len(prompts) - len(tasks))

        async def _generate_one(idx, key, prompt):
            response = await self.generate(prompt, max_tokens, temperature, system)
            results[idx] = response
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "response": response}) + "\n")
            return idx

        await asyncio.gather(*[_generate_one(i, k, p) for i, k, p in tasks])
        return results


class MultiClient:
    """Use different LLM clients for different tasks.

    Example:
        mc = MultiClient(
            generation=LLMClient.from_openai_compatible(model="qwen/qwen3-coder-next", ...),
            refactoring=LLMClient.from_anthropic(model="claude-sonnet-4-20250514"),
        )
        mc.generation.generate(...)  # cheap, fast
        mc.refactoring.generate(...)  # high quality
    """

    def __init__(
        self,
        generation: Optional[LLMClient] = None,
        refactoring: Optional[LLMClient] = None,
        classification: Optional[LLMClient] = None,
        default: Optional[LLMClient] = None,
    ):
        self._default = default or generation
        self.generation = generation or self._default
        self.refactoring = refactoring or self._default
        self.classification = classification or self._default

    @classmethod
    def from_single(cls, client: LLMClient) -> "MultiClient":
        """Use one client for everything."""
        return cls(default=client)


# Backward compatibility
AnthropicClient = LLMClient  # old code that imports AnthropicClient still works

"""Shared async OpenRouter client for the validity-pilot pipelines.

Single entry point `chat()` that takes (model, system, user, **kwargs) and
returns the assistant text. Retries on transient errors. Concurrency via an
externally-managed Semaphore (caller controls).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

_KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()


def make_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=_KEY)


async def chat(client, model: str, system: str, user: str,
               *, temperature: float = 0.0, max_tokens: int = 2000,
               retries: int = 3, retry_delay: float = 2.0) -> str:
    """One chat call, retrying on transient errors. Returns the text content."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
    raise RuntimeError(f"chat({model}) failed after {retries} retries: {last_err}")


async def chat_few_shot(client, model: str, system: str,
                        few_shots: list[tuple[str, str]],
                        user: str, *,
                        temperature: float = 0.0, max_tokens: int = 2000,
                        retries: int = 3, retry_delay: float = 2.0) -> str:
    """Chat with system + a list of (user, assistant) few-shot turns + final user."""
    messages = [{"role": "system", "content": system}]
    for u, a in few_shots:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user})
    last_err = None
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
    raise RuntimeError(f"chat({model}) failed after {retries} retries: {last_err}")

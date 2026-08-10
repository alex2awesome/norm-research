#!/usr/bin/env python3
"""Thin wrapper around run_openrouter.py for the AoPS smoke run.

This bypasses the `provider.require_parameters=true` + `reasoning.exclude=true`
constraints in `call_openrouter` (those cause OpenRouter to 404 for the
qwen/qwen3-235b-a22b-2507 model, since no provider currently advertises
support for the exact reasoning-disable parameters).

Usage: python run_openrouter_aops.py --n 10 --run-name aops_pass1 --seed 42
"""
import os, sys, json, urllib.request

# Patch the call_openrouter implementation before importing main.
import run_openrouter as base

_orig_call = base.call_openrouter
def call_openrouter_relaxed(api_key, prompt, max_tokens=16384, temperature=0.0):
    body = {
        "model": base.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stanford-research.norm-research",
            "X-Title": "norm-research aops smoke",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())

base.call_openrouter = call_openrouter_relaxed

# Force the AoPS task + Qwen3-235B model
os.environ.setdefault("OR_MODEL", "qwen/qwen3-235b-a22b-2507")
base.MODEL = os.environ["OR_MODEL"]

# Inject --task aops_forum if not already there
if "--task" not in sys.argv:
    sys.argv += ["--task", "aops_forum"]

base.main()

#!/usr/bin/env python3
"""Run the frozen family-scale semantic alignments over the z.ai GLM subscription API.

This is an independent-instrument replication of the Sonnet alignment pass.
It consumes already compiled, metric-text-only requests and writes raw model
responses plus a provenance manifest. Scoring remains a separate CPU step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


DEFAULT_MODEL = "glm-5"
ENDPOINT_URL = "https://api.z.ai/api/anthropic/v1/messages"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_response_is_degenerate(raw: str) -> bool:
    """Return True if the response is empty/whitespace-only or highly repetitive."""
    stripped = raw.strip()
    if not stripped:
        return True
    counts: dict[str, int] = {}
    for ch in stripped:
        counts[ch] = counts.get(ch, 0) + 1
    max_count = max(counts.values())
    return max_count / len(stripped) > 0.60


def call_glm_api(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: str,
) -> tuple[str, str | None]:
    """Call the Anthropic-compatible z.ai endpoint, returning (text, stop_reason)."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    req = urllib.request.Request(ENDPOINT_URL, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    stop_reason = body.get("stop_reason")
    return text, stop_reason


def call_with_retries(
    *,
    request: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: str,
    request_file: str,
) -> tuple[str, str | None]:
    """Call the API with fixed-temperature retries on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            text, stop_reason = call_glm_api(
                system_prompt=str(request["system_prompt"]),
                user_prompt=str(request["user_prompt"]),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
            if not text or not text.strip():
                raise ValueError("empty text content in API response")
            return text, stop_reason
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
            last_exc = exc
            if attempt < 5:
                time.sleep(4 * attempt)
    raise RuntimeError(f"failed to fetch response for {request_file} after 5 attempts: {last_exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-file", default=os.path.expanduser("~/.z-ai-api-key.txt"))
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.requests_dir.glob("*_request.json")
        if not path.name.startswith("._")
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no alignment request JSON files found")
    requests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, request in zip(paths, requests):
        if request.get("schema") != "metric-seam.three-fleet-semantic-alignment-request.v1":
            raise ValueError(f"unexpected request schema: {path}")
        if request.get("status") != "compiled_for_exactly_one_alignment_call":
            raise ValueError(f"request is not frozen pre-call: {path}")

    api_key = Path(args.key_file).read_text(encoding="utf-8").strip()

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                call_with_retries,
                request=request,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                api_key=api_key,
                request_file=path.name,
            )
            for path, request in zip(paths, requests)
        ]
        results = [f.result() for f in futures]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_texts = []
    for path, request, (raw, stop_reason) in zip(paths, requests, results):
        raw_texts.append(raw)
        out_path = args.output_dir / path.name.replace("_request.json", "_raw.txt")
        out_path.write_text(raw, encoding="utf-8")
        rows.append({
            "request_file": path.name,
            "request_sha256": request["request_sha256"],
            "source_request_model": request["model"],
            "raw_file": out_path.name,
            "raw_sha256": sha256(out_path),
            "finish_reason": stop_reason,
            "degenerate": raw_response_is_degenerate(raw),
        })

    n_degenerate = sum(1 for r in rows if r["degenerate"])
    n = len(rows)
    if n_degenerate > 0.20 * n:
        raise RuntimeError(
            f"degenerate response rate too high: {n_degenerate}/{n} ({n_degenerate / n:.2%})"
        )

    manifest = {
        "schema": "metric-seam.semantic-alignment-glm-run.v1",
        "status": "raw_responses_complete",
        "host": socket.gethostname(),
        "model": args.model,
        "instrument": "glm_subscription_api",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": n,
        "n_degenerate": n_degenerate,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "claim_limits": [
            "This is an independent open-model semantic-alignment instrument, not external ground truth.",
            "Raw outputs are scored later against the same frozen structural contract.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "n": n,
        "n_degenerate": n_degenerate,
        "output": str(args.output_dir),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

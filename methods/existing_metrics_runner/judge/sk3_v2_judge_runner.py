#!/usr/bin/env python3
"""v2 judge runner — model-agnostic via env vars.

Env vars:
  PROMPT_DIR        relative path under REPO containing <key>.txt files
  RESPONSE_DIR      relative path under REPO for outputs
  MODEL_DIR         FULL absolute path to model dir (e.g. /lfs/.../snapshots/<sha>)
                    If omitted, falls back to Llama-3.3-70B-Instruct-FP8 latest snapshot.
  KV_CACHE_DTYPE    optional: "auto", "fp8", "fp8_e4m3", "fp8_e5m2" (default auto)
  TEMP              sampling temperature (default 0.0)
  MAX_TOKENS        per-response cap (default 2000)
  MAX_MODEL_LEN     context window (default 16384)
  BATCH_FLUSH       flush every N prompts (default 200)
  TP_SIZE           tensor parallel size (default 1)
  GPU_MEM_UTIL      gpu memory utilization (default 0.85)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
PROMPT_DIR = REPO / os.environ.get("PROMPT_DIR", "")
RESPONSE_DIR = REPO / os.environ.get("RESPONSE_DIR", "")
RESPONSE_EXT = os.environ.get("RESPONSE_EXT", ".json")
TEMP = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
BATCH_FLUSH = int(os.environ.get("BATCH_FLUSH", "200"))
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")

# Model path: explicit MODEL_DIR overrides; else fallback to Llama-FP8 latest
_MODEL_DIR_ENV = os.environ.get("MODEL_DIR", "")
if _MODEL_DIR_ENV:
    MODEL_DIR = _MODEL_DIR_ENV
else:
    MODEL_BASE = ("/lfs/skampere3/0/shared_hf_cache/"
                  "models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots")
    MODEL_DIR = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]


def split_prompt(text):
    if "\n=== USER ===\n" in text:
        sys_msg, user_msg = text.split("\n=== USER ===\n", 1)
        return sys_msg.strip(), user_msg.strip()
    return "", text


def main():
    print(f"=== v2 judge runner ===", flush=True)
    print(f"  MODEL_DIR:   {MODEL_DIR}", flush=True)
    print(f"  PROMPT_DIR:  {PROMPT_DIR}", flush=True)
    print(f"  RESPONSE_DIR: {RESPONSE_DIR}", flush=True)
    print(f"  KV_CACHE_DTYPE={KV_CACHE_DTYPE}  TP={TP_SIZE}  "
          f"GPU_MEM_UTIL={GPU_MEM_UTIL}  MAX_LEN={MAX_MODEL_LEN}", flush=True)
    print(f"  flush every {BATCH_FLUSH} prompts; temp={TEMP}", flush=True)

    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    # If PROMPT_LIST_FILE is set, use it (ordered list of keys, one per line);
    # else fall back to alphabetical glob of PROMPT_DIR.
    PROMPT_LIST_FILE = os.environ.get("PROMPT_LIST_FILE", "")
    if PROMPT_LIST_FILE and Path(PROMPT_LIST_FILE).exists():
        with open(PROMPT_LIST_FILE) as _f:
            ordered_keys = [line.strip() for line in _f if line.strip()]
        prompt_files = [PROMPT_DIR / f"{k}.txt" for k in ordered_keys
                        if (PROMPT_DIR / f"{k}.txt").exists()]
        print(f"  PROMPT_LIST_FILE: {PROMPT_LIST_FILE} ({len(ordered_keys)} keys, "
              f"{len(prompt_files)} files found)", flush=True)
    else:
        prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
    to_run = [pf for pf in prompt_files
              if not (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).exists()]
    print(f"  {len(prompt_files)} total; {len(to_run)} to run "
          f"({len(prompt_files)-len(to_run)} cached)", flush=True)
    if not to_run:
        print("nothing to do")
        return

    print("loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=MODEL_DIR,
        dtype="auto",
        tensor_parallel_size=TP_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
        trust_remote_code=True,
        # Force text-only on multimodal models (e.g., Qwen3.5-MoE)
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )
    if KV_CACHE_DTYPE != "auto":
        llm_kwargs["kv_cache_dtype"] = KV_CACHE_DTYPE
    llm = LLM(**llm_kwargs)
    # Anti-loop penalties — defaults 0 so this is a no-op unless set via env.
    PRES_PEN = float(os.environ.get("PRESENCE_PENALTY", "0.0"))
    FREQ_PEN = float(os.environ.get("FREQUENCY_PENALTY", "0.0"))
    # Reasoning budget (Shuyo vLLM patch) — defaults None = disabled.
    RB_MAX = os.environ.get("REASONING_BUDGET_MAX_TOKENS")
    RB_START = os.environ.get("REASONING_BUDGET_START_TOKENS", "100")
    RB_END_ID = os.environ.get("REASONING_END_TOKEN_ID")
    sp_kwargs = dict(temperature=TEMP, max_tokens=MAX_TOKENS,
                     presence_penalty=PRES_PEN, frequency_penalty=FREQ_PEN,
                     seed=42 if TEMP > 0 else None)
    if RB_MAX and RB_END_ID:
        sp_kwargs["reasoning_budget_max_tokens"] = int(RB_MAX)
        sp_kwargs["reasoning_budget_start_tokens"] = int(RB_START)
        sp_kwargs["reasoning_end_token_id"] = int(RB_END_ID)
    sampling = SamplingParams(**sp_kwargs)
    print(f"  PRES_PEN={PRES_PEN}, FREQ_PEN={FREQ_PEN}", flush=True)
    if RB_MAX and RB_END_ID:
        print(f"  REASONING_BUDGET: max={RB_MAX}, start={RB_START}, end_token_id={RB_END_ID}", flush=True)

    # ============================================================
    # Pre-flight: tokenize every prompt; truncate user message if it would
    # push input + reserved output beyond MAX_MODEL_LEN. We reserve enough
    # context for MAX_TOKENS of output so vLLM never raises VLLMValidationError.
    # ============================================================
    tokenizer = llm.get_tokenizer()
    OUTPUT_RESERVE = MAX_TOKENS + 256  # output budget + small safety margin
    MAX_INPUT_TOKENS = MAX_MODEL_LEN - OUTPUT_RESERVE
    print(f"  prompt length cap: {MAX_INPUT_TOKENS} input tokens "
          f"(MAX_MODEL_LEN={MAX_MODEL_LEN} - reserve {OUTPUT_RESERVE})", flush=True)

    # If qwen3 thinking mode is enabled, the chat template adds extra tokens —
    # pass that through to apply_chat_template so token count matches vLLM exactly.
    _enable_think = (("qwen3" in MODEL_DIR.lower() or "qwen-3" in MODEL_DIR.lower())
                     and os.environ.get("ENABLE_THINKING", "false").lower() in ("1", "true", "yes"))

    def _count_chat_tokens(sys_msg, user_msg):
        msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else [])
        msgs.append({"role": "user", "content": user_msg})
        kwargs = {}
        if _enable_think:
            kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        try:
            return len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kwargs))
        except TypeError:
            # Older tokenizers: chat_template_kwargs goes inline
            return len(tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                enable_thinking=_enable_think))
        except Exception:
            return len(tokenizer.encode((sys_msg or "") + user_msg))

    def cap_user_msg(sys_msg, user_msg):
        """Return user_msg truncated so the rendered chat fits MAX_INPUT_TOKENS.

        Iterates: count tokens, trim if over, recount, repeat until under cap.
        Trims from the END of user_msg (where dp text lives).
        """
        n_tok = _count_chat_tokens(sys_msg, user_msg)
        if n_tok <= MAX_INPUT_TOKENS:
            return user_msg, n_tok, False
        truncated = user_msg
        orig_tok = n_tok
        # Iterate trim+recount up to 5 times (safety against estimate-error)
        for _ in range(5):
            target_overage = n_tok - MAX_INPUT_TOKENS
            cut_chars = int(target_overage * 5.0) + 500  # generous safety margin
            new_len = max(len(truncated) - cut_chars, 100)
            truncated = truncated[:new_len]
            n_tok = _count_chat_tokens(sys_msg, truncated)
            if n_tok <= MAX_INPUT_TOKENS: break
        truncated += "\n\n[NOTE: prompt was truncated to fit context window]"
        return truncated, orig_tok, True

    # Process in flush-sized sub-batches
    skipped_too_long = 0
    for start in range(0, len(to_run), BATCH_FLUSH):
        sub = to_run[start:start + BATCH_FLUSH]
        messages_list = []
        kept = []
        for pf in sub:
            sys_msg, user_msg = split_prompt(pf.read_text())
            user_suffix = os.environ.get("USER_SUFFIX", "")
            if user_suffix:
                user_msg = user_msg + "\n\n" + user_suffix
            user_msg, orig_tok, was_truncated = cap_user_msg(sys_msg, user_msg)
            if was_truncated:
                print(f"  TRUNCATED {pf.stem}: orig={orig_tok} > cap={MAX_INPUT_TOKENS}", flush=True)
            msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else [])
            msgs.append({"role": "user", "content": user_msg})
            messages_list.append(msgs)
            kept.append(pf)
        print(f"\nbatch {start//BATCH_FLUSH + 1} / "
              f"{(len(to_run)+BATCH_FLUSH-1)//BATCH_FLUSH}: "
              f"{len(messages_list)} prompts", flush=True)
        chat_kwargs = {}
        if "qwen3" in MODEL_DIR.lower() or "qwen-3" in MODEL_DIR.lower():
            enable_think = os.environ.get("ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_think}
        outputs = llm.chat(messages_list, sampling, use_tqdm=True, **chat_kwargs)
        for pf, out in zip(kept, outputs):
            text = out.outputs[0].text.strip()
            (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).write_text(text)
        print(f"  flushed {len(kept)} responses (total written so far: "
              f"{len(list(RESPONSE_DIR.glob('*' + RESPONSE_EXT)))})", flush=True)

    # ============================================================
    # Retry pass: re-generate any responses that don't smart-parse.
    # Higher temperature on retries to escape deterministic dead-ends.
    # Controlled by MAX_RETRIES env var (default 2).
    # ============================================================
    import re as _re
    def _fix_bad_escapes(raw):
        # Fix invalid JSON backslash escapes (e.g. \ast, \mathbb from raw LaTeX)
        return _re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', raw)

    def _try_parse_one(candidate):
        for variant in [candidate, _fix_bad_escapes(candidate)]:
            try:
                obj = json.loads(variant)
                if "results" in obj and len(obj["results"]) >= 5: return obj
            except: pass
        return None

    def _smart_parse(raw):
        raw = raw.strip()
        # Try fenced code block first
        m = _re.search(r"```(?:json)?\s*\n(.*?)```", raw, _re.S)
        if m:
            obj = _try_parse_one(m.group(1).strip())
            if obj: return obj
        # If </think> present, try the suffix
        if "</think>" in raw:
            suffix = raw.rsplit("</think>", 1)[1].strip()
            starts = [i for i, c in enumerate(suffix) if c == "{"]
            for start in starts[:3]:
                end = suffix.rfind("}")
                while end > start:
                    obj = _try_parse_one(suffix[start:end+1])
                    if obj: return obj
                    end = suffix.rfind("}", start, end)
        # General fallback: try both ends of file's { positions
        starts = [i for i, c in enumerate(raw) if c == "{"]
        if not starts: raise ValueError("no {")
        candidates = list(starts[-5:]) + list(starts[:5])
        seen = set()
        for start in candidates:
            if start in seen: continue
            seen.add(start)
            end = raw.rfind("}")
            while end > start:
                obj = _try_parse_one(raw[start:end+1])
                if obj: return obj
                end = raw.rfind("}", start, end)
        raise ValueError("no parseable JSON")

    MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
    for attempt in range(MAX_RETRIES):
        failed = []
        for pf in to_run:
            rf = RESPONSE_DIR / (pf.stem + RESPONSE_EXT)
            if not rf.exists():
                failed.append(pf); continue
            try: _smart_parse(rf.read_text())
            except: failed.append(pf)
        if not failed:
            print(f"  retry pass {attempt+1}: all responses parse cleanly ✅", flush=True)
            break
        print(f"\n=== Retry pass {attempt+1}/{MAX_RETRIES}: "
              f"{len(failed)} files don't parse, regenerating... ===", flush=True)
        # Higher temp on retries to escape deterministic loops
        retry_temp = 0.3 + 0.2 * attempt
        retry_sampling = SamplingParams(temperature=retry_temp, max_tokens=MAX_TOKENS,
                                         presence_penalty=PRES_PEN,
                                         frequency_penalty=FREQ_PEN,
                                         seed=100 + attempt)
        print(f"  retry temp={retry_temp}, seed={100+attempt}", flush=True)
        # Process failed prompts in same flush-sized batches
        for start in range(0, len(failed), BATCH_FLUSH):
            sub = failed[start:start + BATCH_FLUSH]
            messages_list = []
            for pf in sub:
                sys_msg, user_msg = split_prompt(pf.read_text())
                user_suffix = os.environ.get("USER_SUFFIX", "")
                if user_suffix:
                    user_msg = user_msg + "\n\n" + user_suffix
                msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else [])
                msgs.append({"role": "user", "content": user_msg})
                messages_list.append(msgs)
            outputs = llm.chat(messages_list, retry_sampling, use_tqdm=True, **chat_kwargs)
            for pf, out in zip(sub, outputs):
                text = out.outputs[0].text.strip()
                (RESPONSE_DIR / (pf.stem + RESPONSE_EXT)).write_text(text)
            print(f"  retry batch {start//BATCH_FLUSH + 1}: "
                  f"{len(sub)} prompts regenerated", flush=True)

    print(f"=== DONE ===", flush=True)


if __name__ == "__main__":
    main()

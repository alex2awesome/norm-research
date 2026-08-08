"""Standalone CPU-only diagnostic for FAILURE 1 (Seed-OSS-36B-Instruct) and FAILURE 2
(gpt-oss-20b/120b). No GPU / no vLLM engine — tokenizer + chat_template only, plus a
filesystem completeness check for the gpt-oss-120b snapshot.

Never modifies the repo. Standalone under outputs/osl_multi/, per protocol.
"""
import json
import os
import sys

os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")

from transformers import AutoTokenizer  # noqa: E402

SEED_OSS = "ByteDance-Seed/Seed-OSS-36B-Instruct"
GPT_OSS_20B = "openai/gpt-oss-20b"
GPT_OSS_120B = "openai/gpt-oss-120b"

TEST_PROMPTS = [
    "TEXT: The committee reviewed the draft policy and unanimously approved it after minor "
    "edits.\nQUESTION: does this text satisfy 'describes a formal approval process'? "
    "Answer YES or NO.",
    "TEXT: I love pizza but hate pineapple on it, it's just wrong.\nQUESTION: does this text "
    "satisfy 'expresses a strong food preference'? Answer YES or NO.",
    "TEXT: The stock market fell 2% today amid inflation fears.\nQUESTION: does this text "
    "satisfy 'contains a joke'? Answer YES or NO.",
    "TEXT: Please find attached the quarterly report for your review.\nQUESTION: does this "
    "text satisfy 'is a formal business communication'? Answer YES or NO.",
    "TEXT: lol that's the funniest thing I've seen all week.\nQUESTION: does this text "
    "satisfy 'expresses amusement'? Answer YES or NO.",
]


def resolve_snapshot(model_id: str) -> str:
    from methods.metric_implementer.vllm_backend import _resolve_model_path
    return _resolve_model_path(model_id)


def section(title):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def diag_tokenizer(model_id: str, label: str):
    section(f"{label}: resolve + tokenizer load")
    path = resolve_snapshot(model_id)
    print(f"resolved path: {path}")
    print(f"isdir: {os.path.isdir(path)}")
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        print(f"file count: {len(files)}")
        print("files:", files)

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        print("tokenizer loaded OK:", type(tok))
    except Exception as e:
        print(f"TOKENIZER LOAD FAILED: {type(e).__name__}: {e}")
        return None, path

    print("bos:", repr(tok.bos_token), "eos:", repr(tok.eos_token), "pad:", repr(tok.pad_token))
    special = getattr(tok, "all_special_tokens", [])
    print("all_special_tokens:", special)

    # Does the model's own chat template accept `enable_thinking`? (the kwarg score_binary_gen
    # passes). Jinja silently ignores unused kwargs -- it will NOT raise, so we check whether the
    # variable is referenced in the template source instead.
    tmpl = None
    try:
        tmpl = tok.chat_template
    except Exception as e:
        print("no .chat_template attr:", e)
    if tmpl:
        print(f"chat_template length: {len(tmpl)} chars")
        print("references 'enable_thinking':", "enable_thinking" in tmpl)
        print("references 'thinking_budget':", "thinking_budget" in tmpl)
        print("references 'reasoning_effort':", "reasoning_effort" in tmpl)
        print("references '<seed:think>':", "<seed:think>" in tmpl)
        print("references '</seed:think>':", "</seed:think>" in tmpl)
        print("references '<|channel|>':", "<|channel|>" in tmpl)
        print("references literal '<think>':", "<think>" in tmpl and "<seed:think>" not in tmpl)

    for thinking_kw in (False, True):
        section(f"{label}: apply_chat_template(enable_thinking={thinking_kw})")
        msgs = [{"role": "user", "content": TEST_PROMPTS[0]}]
        try:
            rendered = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking_kw)
        except TypeError as e:
            print(f"TypeError (kwarg rejected, falls back per backend try/except): {e}")
            rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"--- rendered prompt tail (last 600 chars) ---")
        print(rendered[-600:])
        print(f"--- rendered prompt length: {len(rendered)} chars ---")

    # Seed-OSS-specific: try its actual control kwarg, thinking_budget=0, to see the pre-filled
    # think-block the template inserts.
    if "thinking_budget" in (tmpl or ""):
        section(f"{label}: apply_chat_template(thinking_budget=0)  <- Seed-OSS's REAL toggle")
        msgs = [{"role": "user", "content": TEST_PROMPTS[0]}]
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking_budget=0)
        print(rendered[-800:])

    if "reasoning_effort" in (tmpl or ""):
        section(f"{label}: apply_chat_template(reasoning_effort='low')  <- gpt-oss's REAL toggle")
        msgs = [{"role": "user", "content": TEST_PROMPTS[0]}]
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
        print(rendered[-1200:])
        section(f"{label}: default render (no reasoning_effort kwarg -> defaults to 'medium')")
        rendered2 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(rendered2[-1200:])

    return tok, path


def check_120b_snapshot():
    section("gpt-oss-120b snapshot completeness check (filesystem only, NO download)")
    path = resolve_snapshot(GPT_OSS_120B)
    print(f"resolved path: {path}")
    if not os.path.isdir(path):
        print("snapshot dir does not exist / resolution failed -> _resolve_model_path returned"
              " the bare hub id, meaning refs/main or snapshots/<rev> is missing entirely.")
        return
    files = sorted(os.listdir(path))
    print(f"snapshot dir listing ({len(files)} entries): {files}")
    have_tokenizer = any(f.startswith("tokenizer") for f in files)
    have_weights = any(f.startswith("model") and f.endswith(".safetensors") for f in files)
    have_index = any(f == "model.safetensors.index.json" for f in files)
    print(f"has tokenizer.json/tokenizer_config.json: {have_tokenizer}")
    print(f"has model-*.safetensors shard(s): {have_weights}")
    print(f"has model.safetensors.index.json: {have_index}")

    blobs_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "blobs")
    if os.path.isdir(blobs_dir):
        incomplete = [f for f in os.listdir(blobs_dir) if f.endswith(".incomplete")]
        complete = [f for f in os.listdir(blobs_dir) if not f.endswith(".incomplete")]
        inc_bytes = sum(os.path.getsize(os.path.join(blobs_dir, f)) for f in incomplete)
        comp_bytes = sum(os.path.getsize(os.path.join(blobs_dir, f)) for f in complete)
        print(f"blobs/: {len(complete)} complete files ({comp_bytes/1e9:.2f} GB), "
              f"{len(incomplete)} .incomplete files ({inc_bytes/1e9:.2f} GB)")
        print("VERDICT: snapshot is INCOMPLETE -- download stalled mid-transfer "
              "(weights + tokenizer never landed)." if incomplete or not have_weights
              else "VERDICT: snapshot looks complete.")


if __name__ == "__main__":
    print("PYTHON:", sys.executable)
    print("HOME:", os.environ.get("HOME"))
    print("HF_HOME:", os.environ.get("HF_HOME"))

    diag_tokenizer(SEED_OSS, "SEED-OSS-36B-INSTRUCT")
    diag_tokenizer(GPT_OSS_20B, "GPT-OSS-20B")
    check_120b_snapshot()

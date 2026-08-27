"""
Embed all rubrics from the 2,000-pair dedup eval with NV-Embed-v2 (or any
instruction-tuned embedding model), compute pairwise cosine, and correlate
against the LLM verdict.

Reports:
  - Spearman + Pearson between embedding cosine and verdict ordinal
  - Per-verdict cosine distribution
  - ROC-style AUC for the duplicate/paraphrase vs related/different binary

Usage:
  CUDA_VISIBLE_DEVICES=2 python embed_correlation_nvembed.py \\
    --eval-jsonl outputs/dedup_eval/creative_writing_2k.jsonl \\
    --model nvidia/NV-Embed-v2
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")

# Default instruction for the construct-similarity embedding
DEFAULT_INSTRUCTION = "Given a writing-rubric item, represent the underlying evaluation construct so that two items measuring the same construct have nearby embeddings."


def load_eval_pairs(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("verdict") in ("duplicate", "paraphrase", "related", "different"):
                    rows.append(r)
            except Exception:
                pass
    return rows


def collect_unique_rubrics(pairs: list[dict]) -> dict:
    """Map rubric_key -> text."""
    out = {}
    for p in pairs:
        if p['a_key'] not in out:
            out[p['a_key']] = f"{p['a_name']}\n{p['a_description']}"
        if p['b_key'] not in out:
            out[p['b_key']] = f"{p['b_name']}\n{p['b_description']}"
    return out


def embed_with_nemotron(model_path: str, texts: list[str], instruction: str,
                         batch_size: int = 8, max_length: int = 512) -> np.ndarray:
    """Load llama-embed-nemotron-8b via SentenceTransformer.
    Embeddings are L2-normalized via the model's Normalize module."""
    from sentence_transformers import SentenceTransformer

    print(f"loading {model_path}...")
    # Per nemotron README: bf16 + padding_side='left'; FA2 if available else eager.
    try:
        model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
            tokenizer_kwargs={"padding_side": "left"},
        )
    except Exception as e:
        print(f"  FA2 failed ({e}); retrying with eager attention")
        model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": "eager", "torch_dtype": "bfloat16"},
            tokenizer_kwargs={"padding_side": "left"},
        )
    model.max_seq_length = max_length
    print(f"  loaded; max_seq_length={model.max_seq_length}")

    prompt = f"Instruct: {instruction}\nQuery: "
    embs = model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # SentenceTransformer with Normalize module already L2-normalizes; be safe.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default=str(ROOT / "outputs/dedup_eval/creative_writing_2k.jsonl"))
    ap.add_argument("--model", default="/lfs/skampere3/0/shared_hf_cache/models--nvidia--llama-embed-nemotron-8b/snapshots/1acaf42b890bafa464ef9a58d1c0db0dd26120d4")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out-csv", default=str(ROOT / "outputs/dedup_eval/nv_embed_correlation.csv"))
    args = ap.parse_args()

    pairs = load_eval_pairs(Path(args.eval_jsonl))
    print(f"loaded {len(pairs):,} valid pairs")
    if not pairs:
        sys.exit("no pairs found")

    rubrics = collect_unique_rubrics(pairs)
    keys = list(rubrics.keys())
    texts = [rubrics[k] for k in keys]
    print(f"unique rubrics to embed: {len(keys):,}")

    embs = embed_with_nemotron(args.model, texts, args.instruction,
                                batch_size=args.batch_size, max_length=args.max_length)
    print(f"embeddings: shape={embs.shape}")
    key_to_emb = dict(zip(keys, embs))

    # Compute NV-Embed cosine for each pair
    rows = []
    for p in pairs:
        if p['a_key'] not in key_to_emb or p['b_key'] not in key_to_emb:
            continue
        nv_cos = float(np.dot(key_to_emb[p['a_key']], key_to_emb[p['b_key']]))
        rows.append({
            "specificity": p.get("specificity"),
            "cosine_zone": p.get("cosine_zone"),
            "ada_cos": float(p.get("cosine", 0.0)),
            "nv_cos": nv_cos,
            "verdict": p.get("verdict"),
        })

    df = pd.DataFrame(rows)
    verdict_ord = {"duplicate": 0, "paraphrase": 1, "related": 2, "different": 3}
    df["verdict_ord"] = df["verdict"].map(verdict_ord)

    print(f"\n=== Correlation: NV-Embed cosine vs verdict_ord ===")
    from scipy.stats import spearmanr, pearsonr
    rho_s, p_s = spearmanr(df["nv_cos"], df["verdict_ord"])
    rho_p, p_p = pearsonr(df["nv_cos"], df["verdict_ord"])
    print(f"  Spearman: rho={rho_s:.3f}  p={p_s:.4f}")
    print(f"  Pearson:  rho={rho_p:.3f}  p={p_p:.4f}")
    print("  (negative = higher cosine -> more 'same'; expected)")

    print(f"\n=== Correlation: text-embedding-3-small cosine vs verdict_ord (baseline) ===")
    rho_s2, _ = spearmanr(df["ada_cos"], df["verdict_ord"])
    rho_p2, _ = pearsonr(df["ada_cos"], df["verdict_ord"])
    print(f"  Spearman: rho={rho_s2:.3f}")
    print(f"  Pearson:  rho={rho_p2:.3f}")

    print(f"\n=== NV-Embed cosine by verdict ===")
    print(df.groupby("verdict")["nv_cos"].describe().round(3))

    # AUC for "same" (duplicate/paraphrase) vs "not same" (related/different)
    df["is_same"] = df["verdict"].isin(["duplicate", "paraphrase"]).astype(int)
    if df["is_same"].sum() > 0 and df["is_same"].sum() < len(df):
        from sklearn.metrics import roc_auc_score
        auc_nv = roc_auc_score(df["is_same"], df["nv_cos"])
        auc_ada = roc_auc_score(df["is_same"], df["ada_cos"])
        print(f"\n=== AUC for 'same' (duplicate/paraphrase) ===")
        print(f"  NV-Embed cos:                  AUC={auc_nv:.3f}")
        print(f"  text-embedding-3-small cos:    AUC={auc_ada:.3f}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nsaved per-pair scores -> {args.out_csv}")


if __name__ == "__main__":
    main()

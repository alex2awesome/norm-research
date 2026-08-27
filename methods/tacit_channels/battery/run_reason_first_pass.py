"""W1b two-stage pass runner — reason-first rescoring + verbalized confidence.

ONE engine load serves both two-stage variants:

  reason_first — stage 1: the model writes 2-3 sentences of criterion-relevant reasoning
    (REASON_FIRST_INSTR replaces the one-word instruction); stage 2: the SAME teacher-forced
    readout with the model's own rationale interposed (assemble_reason_first_tf). Output grid
    arm_id "name_reason_first" serves P-INTF-1 (tf vs reason-first degradation) and P-TOK-1
    (CoT-delta) with two different contrasts against the W1a tf grid.

  confidence — the model's ANSWER is derived from the W1a tf grid (answer = YES iff
    p_yes >= .5; no second judgment pass), then one generation elicits a 0-100 confidence
    (confidence_prompt). Output npz arm_id "name_confidence" holds parsed integers
    (NaN = unparseable; parse rate in scalars). Serves P-STAT-1 v1 (Dienes zero-correlation)
    and prereg P-B5 (verbalized vs |log-odds| proxy validity).

Stage-1 rationales are persisted (jsonl.gz) for articulation-quality analyses (ACT-R
trajectory, P-STAT-3 adjuncts). Canonical form only (cost; declared in the W1b addendum —
the interference/confidence contrasts are within-form).

GPU script; 1 GPU; offline batch. Deterministic: temperature 0.0 both stages.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import parse_bank_cells
from methods.tacit_channels.battery.passes import (
    REASON_FIRST_INSTR, build_single_stage_rows, confidence_prompt,
)

ANSWER_INSTR = "Answer with exactly one word: YES or NO."


def reason_first_generation_prompt(template: str, row: dict, text: str,
                                   max_text_chars: int) -> str:
    base = template.format(rubric=row["content"], text=text[:max_text_chars])
    return base.replace(ANSWER_INSTR, REASON_FIRST_INSTR)


def reason_first_tf_prompt(template: str, row: dict, text: str, rationale: str,
                           max_text_chars: int, max_rationale_chars: int = 800) -> str:
    base = template.format(rubric=row["content"], text=text[:max_text_chars])
    r = " ".join(rationale.strip().split())[:max_rationale_chars]
    return base.replace(ANSWER_INSTR,
                        f"Reasoning: {r}\n\n{ANSWER_INSTR}")


_INT = re.compile(r"\b(\d{1,3})\b")


def parse_confidence(text: str) -> float:
    """First integer in [0,100]; NaN otherwise."""
    for m in _INT.finditer(text or ""):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return float(v)
    return float("nan")


def tf_answers_from_grid(npz_path: str, domain: str) -> dict:
    """{(cell_id, form): bool-array over items} from a W1a grid's tf rows (p_yes >= .5)."""
    d = np.load(npz_path, allow_pickle=True)
    scores = np.asarray(d["scores"])
    out = {}
    for i, s in enumerate(d["meta"]):
        m = json.loads(s)
        if m.get("variant") == "tf" and m.get("domain") == domain:
            if not np.isfinite(scores[i]).all():
                bad = int((~np.isfinite(scores[i])).sum())
                raise ValueError(
                    f"tf grid row ({m['cell_id']},{m['form']}) has {bad} non-finite "
                    "scores — refusing to derive answers (NaN >= .5 would silently "
                    "read as NO)")
            out[(m["cell_id"], m["form"])] = scores[i] >= 0.5
    return out


def run_reason_first(backend, rows, texts, template, max_text_chars, label_token_ids,
                     gen_max_tokens, score_fn=None, gen_fn=None):
    if score_fn is None:
        from methods.tacit_channels.channels.eval.teacher_forced_lora import (
            score_declared_binary_lora)
        score_fn = score_declared_binary_lora
    if gen_fn is None:
        gen_fn = lambda prompts, seed: backend.generate_batch(
            prompts, max_tokens=gen_max_tokens, temperature=0.0, seed=seed)
    scores, meta, rationales = [], [], []
    for row in rows:
        gen_prompts = [reason_first_generation_prompt(template, row, t, max_text_chars)
                       for t in texts]
        outs = gen_fn(gen_prompts, 20260723)
        tf_prompts = [reason_first_tf_prompt(template, row, t, o or "", max_text_chars)
                      for t, o in zip(texts, outs)]
        row_seed = len(meta) * 1009 + 20260713
        vec = score_fn(backend, tf_prompts, pos="YES", neg="NO",
                       expected_token_ids=label_token_ids, seed=row_seed)
        scores.append(np.asarray(vec, float))
        meta.append({"cell_id": row["cell_id"], "arm_id": "name_reason_first",
                     "variant": "reason_first", "form": row["form"],
                     "domain": row.get("domain"),
                     "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest()})
        rationales.append({"cell_id": row["cell_id"], "form": row["form"],
                           "rationales": [o or "" for o in outs]})
    return np.vstack(scores), meta, rationales


def run_confidence(backend, rows, texts, template, max_text_chars, answers,
                   gen_fn=None):
    if gen_fn is None:
        gen_fn = lambda prompts, seed: backend.generate_batch(
            prompts, max_tokens=8, temperature=0.0, seed=seed)
    conf, meta, parse_rates = [], [], []
    for row in rows:
        key = (row["cell_id"], row["form"])
        if key not in answers:
            continue
        ans = answers[key]
        prompts = []
        for t, a in zip(texts, ans):
            base = template.format(rubric=row["content"], text=t[:max_text_chars])
            prompts.append(confidence_prompt(base, "YES" if a else "NO"))
        outs = gen_fn(prompts, 20260723)
        vals = np.array([parse_confidence(o) for o in outs], float)
        conf.append(vals)
        parse_rates.append(float(np.isfinite(vals).mean()))
        meta.append({"cell_id": row["cell_id"], "arm_id": "name_confidence",
                     "variant": "confidence", "form": row["form"],
                     "domain": row.get("domain"),
                     "parse_rate": parse_rates[-1]})
    return (np.vstack(conf) if conf else np.zeros((0, len(texts)))), meta, parse_rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-adapter", default=None)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--packet-root", required=True)
    ap.add_argument("--partitions", default="tacit_breadth_search")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--readout-template", required=True)
    ap.add_argument("--max-text-chars", type=int, required=True)
    ap.add_argument("--yes-id", type=int, required=True)
    ap.add_argument("--no-id", type=int, required=True)
    ap.add_argument("--stages", default="reason_first,confidence")
    ap.add_argument("--tf-grid", default=None,
                    help="W1a grid npz for this (model,adapter) — REQUIRED for confidence")
    ap.add_argument("--forms", default="canonical")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--limit-cells", type=int, default=None)
    ap.add_argument("--gen-max-tokens", type=int, default=160)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--intervention-tag", default="base")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--upstream-sha", default=None)
    args = ap.parse_args()

    stages = set(args.stages.split(","))
    if "confidence" in stages and not args.tf_grid:
        raise SystemExit("--tf-grid required for the confidence stage")
    from methods.tacit_channels.channels.eval.teacher_forced_lora import (
        check_upstream_drift, upstream_source_sha256)
    if args.upstream_sha:
        check_upstream_drift(args.upstream_sha)

    cells = parse_bank_cells(args.bank)
    cells = {cid: c for cid, c in cells.items() if c.get("domain") == args.domain}
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = {cid: c for cid, c in cells.items() if cid in wanted}
    if args.limit_cells:
        cells = {cid: cells[cid] for cid in sorted(cells)[:args.limit_cells]}
    rows = build_single_stage_rows(cells, ("tf",), forms=tuple(args.forms.split(",")))
    for r in rows:
        r["domain"] = args.domain
    print(f"two-stage plan: {len(rows)} name rows x {args.stages}")

    items = _apparatus.load_domain_items(
        args.packet_root, args.domain, partitions=args.partitions.split(","))
    texts = items["texts"]
    template = Path(args.readout_template).read_text()
    label_token_ids = {"YES": args.yes_id, "NO": args.no_id}

    from methods.tacit_channels.channels.eval.score_with_adapter import build_backend
    backend = build_backend(args.model, args.lora_adapter, args.tp_size,
                            args.max_model_len, args.gpu_mem_util, args.fake)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(
        f"{args.model}::{args.lora_adapter}::{args.intervention_tag}::w1b"
        .encode()).hexdigest()[:16]

    if "reason_first" in stages:
        matrix, meta, rationales = run_reason_first(
            backend, rows, texts, template, args.max_text_chars, label_token_ids,
            args.gen_max_tokens)
        if not np.isfinite(matrix).all():
            raise SystemExit("REFUSING to write reason_first grid: non-finite scores")
        out = out_dir / f"grid_{args.domain}_w1b_reason_{tag}_rep0.npz"
        np.savez_compressed(
            out, scores=matrix, meta=np.array([json.dumps(m) for m in meta], dtype=object),
            model=args.model, lora_adapter=str(args.lora_adapter),
            intervention_tag=args.intervention_tag,
            upstream_declared_binary_sha256=upstream_source_sha256(),
            readout="teacher_forced_declared_labels(lora-fork,reason-first)")
        with gzip.open(out_dir / f"rationales_{args.domain}_{tag}.jsonl.gz", "wt") as f:
            for r in rationales:
                f.write(json.dumps(r) + "\n")
        print(f"reason_first: wrote {matrix.shape} -> {out}")

    if "confidence" in stages:
        answers = tf_answers_from_grid(args.tf_grid, args.domain)
        conf, cmeta, rates = run_confidence(
            backend, rows, texts, template, args.max_text_chars, answers)
        out = out_dir / f"confidence_{args.domain}_w1b_{tag}_rep0.npz"
        np.savez_compressed(
            out, scores=conf, meta=np.array([json.dumps(m) for m in cmeta], dtype=object),
            model=args.model, lora_adapter=str(args.lora_adapter),
            intervention_tag=args.intervention_tag, tf_grid=str(args.tf_grid),
            mean_parse_rate=float(np.mean(rates)) if rates else 0.0,
            readout="verbalized_confidence_0_100")
        print(f"confidence: wrote {conf.shape} (parse rate "
              f"{float(np.mean(rates)) if rates else 0:.3f}) -> {out}")


if __name__ == "__main__":
    main()

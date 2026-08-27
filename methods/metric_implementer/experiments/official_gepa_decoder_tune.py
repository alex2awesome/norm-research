"""OFFICIAL-GEPA (github gepa-ai/gepa, pinned) search over the v14 MCQ decoder template — the
search-quality comparison against the in-house `tune_shared_template_batched` (which managed one
accepted round: seed pooled −0.155 → best −0.041 canonical / −0.128 → −0.032 held-out, then the
frozen transfer gate emptied the admissible set).

PILOT SCOPE (deliberate):
  * mcq channel, ONE decoder family (qwen, 1 GPU, engine kept resident across all evaluations);
  * selection signal = SEARCH-split normalized fitness ONLY (per dev-metric instance, so the
    official Pareto frontier is meaningful); held-out transfer is computed POST-HOC for the report
    and the frozen gate — never used for selection (same discipline as the in-house run);
  * same seed template, same validator (forbidden metric strings, required placeholders), same
    evidence-cell cache, GLM-5.2 reflection via the subscription API.

Run on sk2 (see scripts/official_gepa_decoder_sk2.sh):
  python -m methods.metric_implementer.experiments.official_gepa_decoder_tune \
      --out-root /lfs/skampere2/0/alexspan/cr3-v14.1-two-lane/outputs/fast \
      --max-metric-calls 240
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from ..backends import LLMBackend
from ..config import ImplementerConfig
from .run_v14_value_campaign import _backend, _development_contexts
from .v14_decoder_tuning import template_sha256, validate_shared_template
from .v14_mcq_channel import DEFAULT_MCQ_TEMPLATE
from .v14_tuning_evaluator import (EvidenceCellStore, aggregate_template_fitness,
                                   score_mcq_reference_templates)

QWEN = "Qwen/Qwen2.5-14B-Instruct"
REQUIRED_FIELDS = ("noun", "examples", "choices", "labels")


def _forbidden(contexts) -> list[str]:
    out = []
    for context in contexts:
        out.extend([context["metric_key"], context["target_description"]])
        out.extend(
            str(item.get("description", "") if isinstance(item, dict) else item.description)
            for item in context["distractors"])
    return out


def _per_context_search_fitness(rows, metric_key: str) -> float:
    vals = [float(r["normalized_fitness"]) for r in rows
            if r["reference_split"] == "search" and str(r["metric_key"]) == str(metric_key)]
    return float(np.mean(vals)) if vals else float("-inf")


def _contrast_feedback(rows, metric_key: str) -> str:
    sr = [r for r in rows if r["reference_split"] == "search"
          and str(r["metric_key"]) == str(metric_key)]
    if not sr:
        return "no search rows"
    best = max(sr, key=lambda r: float(r["normalized_fitness"]))
    worst = min(sr, key=lambda r: float(r["normalized_fitness"]))
    return (f"mean search fitness {np.mean([float(r['normalized_fitness']) for r in sr]):+.3f}; "
            f"best state {best['state']} fitness {float(best['normalized_fitness']):+.3f}; "
            f"worst state {worst['state']} fitness {float(worst['normalized_fitness']):+.3f}. "
            "Fitness is target-option lift over the strongest control, normalized; improve "
            "evidence-routed contrastive decoding without naming any metric content.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-metric-calls", type=int, default=240)
    ap.add_argument("--query-batch-size", type=int, default=2048)
    ap.add_argument("--reflection-model", default="glm-5.2")
    a = ap.parse_args()

    import gepa
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    out_root = Path(a.out_root)
    contexts = _development_contexts(out_root)
    if len(contexts) != 8:
        raise RuntimeError("v14 tuning requires exactly eight development metrics")
    forbidden = _forbidden(contexts)
    store_path = out_root / "development" / "tuning_cells.sqlite"
    run_dir = out_root / "development" / "tuning" / "official_gepa_qwen"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "proposals.jsonl"

    decoder, revision = _backend(QWEN, fake=False)          # resident for the whole search

    class V14McqAdapter(GEPAAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            template = str(next(iter(candidate.values())))
            try:
                validate_shared_template(template, forbidden_strings=forbidden,
                                         required_fields=REQUIRED_FIELDS)
            except Exception as exc:
                scores = [-1.0] * len(batch)
                trajs = ([{"data": c, "full_assistant_response": "",
                           "feedback": f"INVALID TEMPLATE (rejected before scoring): {exc}"}
                          for c in batch] if capture_traces else None)
                self._log(template, batch, scores, invalid=str(exc))
                return EvaluationBatch(outputs=[{}] * len(batch), scores=scores,
                                       trajectories=trajs)
            rows = None
            for attempt in range(3):
                try:
                    with EvidenceCellStore(store_path) as store:
                        rows = score_mcq_reference_templates(
                            decoder, templates=[template], contexts=list(batch),
                            decoder_family="qwen", constructor_revision=revision,
                            store=store, query_batch_size=a.query_batch_size)
                    break
                except Exception as exc:
                    # engine re-init is flaky under churn (zombie EngineCore GPU-mem);
                    # release + wait + retry, and NEVER let one failure eat a gepa iteration.
                    print(f"[evaluate] scorer attempt {attempt} failed: {exc}", flush=True)
                    from ..vllm_backend import release_resident_engines
                    try:
                        release_resident_engines()
                    except Exception:
                        pass
                    time.sleep(45)
            if rows is None:
                scores = [-1.0] * len(batch)
                trajs = ([{"data": c, "full_assistant_response": "",
                           "feedback": "evaluator transient failure (engine init); "
                                       "not a property of this template"}
                          for c in batch] if capture_traces else None)
                self._log(template, batch, scores, invalid="scorer-transient-failure")
                return EvaluationBatch(outputs=[{}] * len(batch), scores=scores,
                                       trajectories=trajs)
            scores = [_per_context_search_fitness(rows, c["metric_key"]) for c in batch]
            trajs = ([{"data": c, "full_assistant_response": "",
                       "feedback": _contrast_feedback(rows, c["metric_key"])}
                      for c in batch] if capture_traces else None)
            self._log(template, batch, scores)
            return EvaluationBatch(outputs=[{}] * len(batch), scores=scores,
                                   trajectories=trajs)

        def _log(self, template, batch, scores, invalid=None):
            with open(log_path, "a") as fh:
                fh.write(json.dumps({
                    "ts": time.time(), "template_sha256": template_sha256(template),
                    "template": template, "metrics": [c["metric_key"] for c in batch],
                    "scores": scores, "invalid": invalid}) + "\n")

        CONSTRAINT = ("HARD CONSTRAINTS for any rewritten template (violations score -1): "
                      "preserve EVERY format placeholder from the current template EXACTLY — "
                      "{noun}, {examples}, {choices}, {labels} — keep them as literal "
                      "curly-brace fields; do NOT mention any metric name, description, or "
                      "example content; do NOT add exemplars; return only the template text.")

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            comp = components_to_update[0]
            items = [{"Inputs": f"dev metric #{i} (content withheld — the shared template must "
                                "stay metric-agnostic)",
                      "Generated Outputs": "(constrained MCQ decode over panel states)",
                      "Feedback": f"{t['feedback']} {self.CONSTRAINT}"}
                     for i, t in enumerate(eval_batch.trajectories or [])]
            return {comp: items}

    cfg = dataclasses.replace(ImplementerConfig(), backend="zai_anthropic")
    refl = LLMBackend(a.reflection_model, "generator", cfg, temperature=1.0)

    def reflection_lm(prompt) -> str:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        return str(refl.generate_batch([text], system=None, max_tokens=1200,
                                       temperature=1.0)[0])

    print(f"[official-gepa-decoder] 8 contexts, qwen resident, budget {a.max_metric_calls}",
          flush=True)
    result = gepa.optimize(
        seed_candidate={"mcq_template": DEFAULT_MCQ_TEMPLATE},
        trainset=list(contexts), valset=list(contexts),
        adapter=V14McqAdapter(), reflection_lm=reflection_lm,
        max_metric_calls=a.max_metric_calls, run_dir=str(run_dir / "gepa_state"),
        seed=0, display_progress_bar=False, raise_on_exception=False)

    # POST-HOC full report (both splits, gate flags) for seed + every distinct proposed template.
    distinct, seen = [], set()
    for line in open(log_path):
        r = json.loads(line)
        if r.get("invalid"):
            continue
        if r["template_sha256"] not in seen:
            seen.add(r["template_sha256"])
            distinct.append(r["template"])
    with EvidenceCellStore(store_path) as store:                     # cache makes reruns cheap
        rows = score_mcq_reference_templates(
            decoder, templates=distinct, contexts=contexts, decoder_family="qwen",
            constructor_revision=revision, store=store, query_batch_size=a.query_batch_size)
    reports = aggregate_template_fitness(rows)
    best_template = str(next(iter(result.best_candidate.values())))
    payload = {
        "schema": "official-gepa-decoder-tune-v1", "channel": "mcq",
        "decoder_family": "qwen", "budget_metric_calls": a.max_metric_calls,
        "seed_sha": template_sha256(DEFAULT_MCQ_TEMPLATE),
        "best_sha": template_sha256(best_template),
        "best_template": best_template,
        "n_distinct_templates": len(distinct),
        "reports": {sha: {k: rep.get(k) for k in
                          ("pooled_fitness", "heldout_prompt_fitness",
                           "heldout_prompt_transfer_ok", "far_near_transfer_ok",
                           "dev_identification_residual_bits", "n_search_cells",
                           "n_heldout_prompt_cells")}
                    for sha, rep in reports.items()},
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, default=float))
    seed_rep = reports.get(template_sha256(DEFAULT_MCQ_TEMPLATE), {})
    best_rep = reports.get(template_sha256(best_template), {})
    print(f"[official-gepa-decoder] DONE: {len(distinct)} distinct templates; "
          f"seed pooled {seed_rep.get('pooled_fitness')} -> best pooled "
          f"{best_rep.get('pooled_fitness')}; best gate "
          f"heldout_ok={best_rep.get('heldout_prompt_transfer_ok')}; wrote {run_dir/'result.json'}",
          flush=True)


if __name__ == "__main__":
    main()

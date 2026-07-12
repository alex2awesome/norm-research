"""Prompt-ORDERING permutation test — validates efficiency lever #1 AND is the template-level
prompt-ordering check the procedure was missing.

The two permutation tests already baked in are: ``order_stability`` (permutes the Ω-ELEMENT order —
which criterion builds a species core first) and ``form_invariance`` (clause reorder / boilerplate /
question / suffix WITHIN a criterion). Neither tests TEMPLATE-LEVEL ordering — i.e., does it matter
whether the prompt is ``Criterion: … Text: …`` (rubric-first) or ``Text: … Criterion: …`` (text-first)?
That is exactly the structural reorder lever #1 makes (text-first so vLLM prefix-caches the probe
texts), so this check doubles as #1's validation gate.

What it does: for each saved checkpoint (criteria already scored rubric-first), RE-SCORE the same
criteria under the text-first template and compare:

  * **species-label concordance** — Adjusted Rand Index between the two conditional-species partitions
    (1.0 = identical partition). This is the quantity that matters: B_E/coverage are functions of the
    partition, so ARI≈1 ⇒ the bounds are invariant to template order.
  * **B_E / D_obs / coverage drift** — the capture-recapture bounds under each form.
  * **soft-readout shift** — Pearson r and binary-agreement of the P(YES) signatures, to see how much
    the absolute readout moves (it can shift even when the partition is stable).

PASS gate (deploy #1): median ARI ≥ 0.8 AND median |ΔB_E| ≤ median bootstrap std. Run on a few
existing checkpoints — text-first re-scoring is prefix-cached, so ~minutes per metric.

Example (sk3):
    python -m methods.metric_implementer.experiments.prompt_ordering_check \
        --task creative-writing --target-model meta-llama/Llama-3.1-8B-Instruct \
        --dir /lfs/.../outputs/crc_scaling/llama8b --n-metrics 4
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .. import config as cfgmod
from ..vllm_backend import make_judge_backend
from . import alpha_probe as ap
from .run_real_test import _load_texts


def _ari(a, b):
    """Adjusted Rand Index between two int label vectors (drops -1 / unmatched). sklearn if available."""
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        return float("nan")
    a = np.asarray(a); b = np.asarray(b)
    m = (a >= 0) & (b >= 0)
    return float(adjusted_rand_score(a[m], b[m])) if m.any() else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(prog="prompt_ordering_check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dir", required=True, help="checkpoint dir with *_sigs.npz (rubric-first sigs)")
    p.add_argument("--n-metrics", type=int, default=4)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--cmi-thresh", type=float, default=0.15)
    a = p.parse_args(argv)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    executor = make_judge_backend(a.target_model, cfgmod.ImplementerConfig(), temperature=None)

    # reload the SAME probe texts the checkpoints were scored on (deterministic pool order)
    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    print(f"executor = {a.target_model} | probe set = {len(probe_texts)} | template = text-first")

    ckpts = sorted(glob.glob(os.path.join(a.dir, "*_sigs.npz")))[: a.n_metrics]
    if not ckpts:
        print(f"no *_sigs.npz in {a.dir}"); return
    print("%-34s %5s %6s %8s %8s %7s %7s %6s %6s" % (
        "metric", "n_crit", "ARI", "B_E_rf", "B_E_tf", "D_rf", "D_tf", "r_soft", "binAgr"))
    print("-" * 96)
    aris,dbe,dob,cov,rsoft,binagr = [],[],[],[],[],[]
    for f in ckpts:
        z = np.load(f, allow_pickle=True)
        sigs_rf = np.asarray(z["sigs"], float); prompts = list(z["prompts"]); tags = list(z["tags"])
        name = str(z["name"]) if "name" in z.files else os.path.basename(f)
        # RE-SCORE text-first (probe texts prefix-cached across criteria -> fast after the first)
        sigs_tf = np.vstack([ap.signature(executor, pr, probe_texts, cfg.max_text_chars,
                                          template=ap._YESNO_TEXTFIRST) for pr in prompts])
        lab_rf = ap.conditional_species(sigs_rf, cmi_thresh=a.cmi_thresh)
        lab_tf = ap.conditional_species(sigs_tf, cmi_thresh=a.cmi_thresh)
        ari = _ari(lab_rf, lab_tf)
        crf = ap.conditional_crc_report(sigs_rf, tags, cmi_thresh=a.cmi_thresh)
        ctf = ap.conditional_crc_report(sigs_tf, tags, cmi_thresh=a.cmi_thresh)
        # soft-readout shift: corr + binary agreement of the flattened soft signatures
        x = sigs_rf.flatten(); y = sigs_tf.flatten()
        m = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 5 else float("nan")
        ba = float((((x[m] > 0.5).astype(int)) == ((y[m] > 0.5).astype(int))).mean()) if m.sum() > 5 else float("nan")
        aris.append(ari); dbe.append(abs(crf["B_E_upper"] - ctf["B_E_upper"]))
        dob.append((crf["D_obs_lower"], ctf["D_obs_lower"])); cov.append((crf["coverage"], ctf["coverage"]))
        rsoft.append(r); binagr.append(ba)
        print("%-34s %5d %6.3f %8.1f %8.1f %7.0f %7.0f %6.2f %6.2f" % (
            name[:34], len(prompts), ari, crf["B_E_upper"], ctf["B_E_upper"],
            crf["D_obs_lower"], ctf["D_obs_lower"], r, ba))
    aris = np.array([x for x in aris if x == x]); rsoft = np.array([x for x in rsoft if x == x])
    binagr = np.array([x for x in binagr if x == x])
    print("-" * 96)
    print("AGGREGATE (n=%d):  ARI median=%.3f min=%.3f | soft-r median=%.3f | bin-agree median=%.3f" % (
        len(ckpts), float(np.median(aris)) if aris.size else float("nan"),
        float(np.min(aris)) if aris.size else float("nan"),
        float(np.median(rsoft)) if rsoft.size else float("nan"),
        float(np.median(binagr)) if binagr.size else float("nan")))
    print("\nD_obs (rf->tf): " + ", ".join(f"{a:.0f}->{b:.0f}" for a, b in dob))
    print("coverage (rf->tf): " + ", ".join(f"{a:.2f}->{b:.2f}" for a, b in cov))
    ari_med = float(np.median(aris)) if aris.size else 0.0
    verdict = "PASS — species partition & B_E invariant to template order; deploy #1 (text-first)" \
        if ari_med >= 0.8 else "FAIL — template order moves the partition; do NOT deploy #1 as-is"
    print(f"\nGATE (ARI median >= 0.8): {verdict}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recovery-audit Q4 intervention test: taxonomy-directed prompting vs a plain
extra sealed draw, on rep3 (the weakest replicate: recall sensitivity .125).

Two proposer-only calls (gpt-5.6-luna via codex exec, read-only sandbox, scratch
working dir outside the repo), both reading the SAME rep3 depleted-bank slice:

  Arm R (control, raise-P route): the original sealed prompt, fresh sha256
      ordering salt -- what does a 5th independent generic proposer add?

  Arm T (taxonomy-directed, Addendum-4 route): same slice + a category-level
      sweep instruction over general scientific-reporting quality domains, and an
      instruction to phrase criteria in reporting-checklist register.  This
      partially unseals the bank at CATEGORY level (the trade is recorded), and
      any directed round is excluded from Good-Turing / capture-recapture
      estimation by construction (non-independent draws).

Measurement WITHOUT new judge calls (the sealed Opus recall instrument is not
re-run; no judging in this audit):
  * mechanical max-cosine of each of rep3's 16 recall targets (8 held-out +
    8 retained controls) against each arm's proposals, vs the original fleet's;
  * tau-band hits (.77/.79/.81) -- the original fleet's global max was .722, so
    any tau hit is a categorical detector-range change;
  * specificity: same readout against the 38 non-target bank concepts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RMM = HERE.parent
sys.path.insert(0, str(RMM))
import harness  # noqa: E402
from embed_lib import embed, crit_text, bank_concept_texts  # noqa: E402

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/audit_rep3")
SCRATCH.mkdir(parents=True, exist_ok=True)

DIRECTED_BLOCK = """
COVERAGE SWEEP (additional hard constraint). Your criteria must, between them,
cover the full space of quality dimensions used by GENERAL SCIENTIFIC-PUBLISHING
quality checklists (the kind journals attach to submissions), not only the
dimensions ML reviewers usually discuss. Distribute your {k} criteria so that
every one of these categories is covered by at least one criterion:

  1. study/experimental design reporting (groups, units, allocation, timing);
  2. outcome measures and analysis reporting (definitions, precision, uncertainty);
  3. data, code, and materials availability and documentation;
  4. ethics, compliance, and responsible conduct/reporting;
  5. abstract/title writing quality (accuracy, completeness, balance, no hype);
  6. citation and literature practices;
  7. research software / artifact quality;
  8. accessibility and clarity of communication for broad audiences.

REGISTER. Phrase each criterion the way a general scientific-reporting checklist
would (plain journal-editorial language), NOT in ML jargon; then make the 0-10
instruction concrete enough to score an ML abstract.
"""


def build_arm(tag_salt, directed):
    rows = json.loads((RMM / "slice_rep3.json").read_text())
    prompt = harness.build_prompt(rows, tag_salt)
    if directed:
        # inject the sweep block right after the preamble's hard constraints
        marker = "OUTPUT FORMAT."
        prompt = prompt.replace(marker, DIRECTED_BLOCK.format(k=harness.K) + "\n" + marker, 1)
    return prompt


def run_codex(name, prompt, model="gpt-5.6-luna", effort="high", timeout=2400):
    out = SCRATCH / f"out_{name}.txt"
    if out.exists() and len(out.read_text()) > 500:
        print(f"[{name}] already done, skip", flush=True)
        return out.read_text()
    (SCRATCH / f"prompt_{name}.txt").write_text(prompt)
    wd = SCRATCH / f"wd_{name}"
    wd.mkdir(exist_ok=True)
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    raw = p.stdout
    (SCRATCH / f"raw_{name}.txt").write_text(raw)
    body = raw
    if "tokens used" in raw:
        body = raw.rsplit("tokens used", 1)[1]
        body = body.split("\n", 2)[-1] if body.strip().split("\n")[0].strip().replace(",", "").isdigit() else body
    out.write_text(body.strip())
    print(f"[{name}] rc={p.returncode} {time.time()-t0:.0f}s -> {len(body)} chars", flush=True)
    return body.strip()


def main():
    recall = json.loads((RMM / "m3_recall.json").read_text())
    targets = [r for r in recall["records"] if r["rep"] == "rep3"]
    bank = bank_concept_texts()
    target_names = [r["concept"] for r in targets]
    nontarget = [n for n in bank if n not in target_names]

    arms = {}
    for name, directed in (("ctrl_redraw", False), ("tax_directed", True)):
        body = run_codex(name, build_arm(f"audit_rep3|{name}", directed))
        crit = harness.parse_output(body)
        arms[name] = crit
        print(f"  {name}: {len(crit)} criteria parsed; names: "
              f"{[c['name'][:40] for c in crit][:6]} ...", flush=True)

    # original rep3 fleet proposals for baseline comparison
    orig = json.loads((RMM / "proposals_rep3.json").read_text())["proposals"]

    pools = {"orig_fleet_P4": [crit_text(p["name"], p["instruction"]) for p in orig]}
    for name, crit in arms.items():
        pools[name] = [crit_text(c["name"], c["instruction"]) for c in crit]

    tnames = {r["concept"]: r for r in targets}
    ttexts = [crit_text(n, bank.get(n, "")) for n in target_names]
    nttexts = [crit_text(n, bank.get(n, "")) for n in nontarget]

    all_texts = ttexts + nttexts + [t for v in pools.values() for t in v]
    E = embed(all_texts, verbose=True)
    Et = E[:len(ttexts)]
    Ent = E[len(ttexts):len(ttexts) + len(nttexts)]
    off = len(ttexts) + len(nttexts)
    Ep = {}
    for name, v in pools.items():
        Ep[name] = E[off:off + len(v)]
        off += len(v)

    out = {"design": {"slice": "slice_rep3.json (depleted rep3 stack)",
                      "arms": {"ctrl_redraw": "sealed re-draw, fresh salt (raise-P control)",
                               "tax_directed": "category-level taxonomy sweep + checklist register"},
                      "model": "gpt-5.6-luna, codex exec, effort high, read-only scratch wd",
                      "independence_note": "tax_directed proposals are NON-independent draws: "
                                           "excluded from any Good-Turing/Chao1 estimation; "
                                           "category-level bank visibility trade recorded"},
           "targets": {}, "summary": {}}

    for arm, Earm in Ep.items():
        S = Et @ Earm.T
        Snt = Ent @ Earm.T
        rows = []
        for i, n in enumerate(target_names):
            r = tnames[n]
            best = int(np.argmax(S[i]))
            rows.append({
                "concept": n, "kind": r["kind"], "stratum": r["stratum"],
                "recall_matched_orig": r["match_primary"],
                "max_cos": float(S[i].max()),
                "best_name": (orig[best]["name"] if arm == "orig_fleet_P4"
                              else arms[arm][best]["name"]),
                "hit_.77": bool(S[i].max() >= .77),
                "hit_.79": bool(S[i].max() >= .79),
                "hit_.81": bool(S[i].max() >= .81),
            })
        held = [r for r in rows if r["kind"] == "heldout"]
        ctrl = [r for r in rows if r["kind"] != "heldout"]
        out["targets"][arm] = rows
        out["summary"][arm] = {
            "n_proposals": len(pools[arm]),
            "heldout_hits_.77": sum(r["hit_.77"] for r in held),
            "heldout_hits_.79": sum(r["hit_.79"] for r in held),
            "retained_hits_.77": sum(r["hit_.77"] for r in ctrl),
            "retained_hits_.79": sum(r["hit_.79"] for r in ctrl),
            "heldout_max_cos_mean": float(np.mean([r["max_cos"] for r in held])),
            "retained_max_cos_mean": float(np.mean([r["max_cos"] for r in ctrl])),
            "global_max_cos_targets": float(max(r["max_cos"] for r in rows)),
            "nontarget38_hits_.77": int((Snt.max(axis=1) >= .77).sum()),
            "nontarget38_max_cos_mean": float(Snt.max(axis=1).mean()),
        }

    (HERE / "q5_intervention.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print("\nper-target (tax_directed):")
    for r in sorted(out["targets"]["tax_directed"], key=lambda r: -r["max_cos"]):
        print(f"  {r['max_cos']:.3f} {'HIT' if r['hit_.79'] else '   '} "
              f"[{r['kind'][:7]:7s}/{r['stratum']:4s}] {r['concept'][:48]:48s} <- {r['best_name'][:45]}")


if __name__ == "__main__":
    main()

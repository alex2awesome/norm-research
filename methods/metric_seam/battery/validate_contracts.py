"""WS1a central validation of authored construct contracts (runbook 2026-07-10).

Gates (pre-registered, BEFORE freeze):
  1. valid JSON with all 4 keys
  2. construct_definition verbatim-matches the pack's criterion_description
  3. 4-6 cf_probes; each text_pos/text_neg nonempty AND distinct
  4. no probe references labels/scores/metadata/length/dataset artifacts (keyword scan;
     flagged probes are listed for manual read — scan is a net, not a verdict)
  5. discrimination_checks exactly {min_std: 0.05, max_frac_at_mode: 0.85}
A contract passing 1-3+5 with no UNADJUDICATED flags is FROZEN.
v2 (2026-07-10, post external review): keyword flags (gate 4) are now WARNINGS, kept
separate from hard errors, matching the docstring; adjudications live in
contract_flag_adjudications.json ({"<name>": {"probe <i>": "<specific reason benign>"}}),
and an unadjudicated flag blocks the pass. Exit code: 0 all pass, 1 otherwise.
Usage: python3 validate_contracts.py
"""
import json
import pathlib
import re

EL = pathlib.Path(__file__).resolve().parents[3] / \
    "outputs/metric_seam_pilot/battery/effort_ladder"
FORBIDDEN = re.compile(
    r"\b(label|labels|labeled|gold|ground[- ]truth|score distribution|train(ing)? (set|data|items)"
    r"|dataset|metadata|word count|character count|document length|longer texts?|shorter texts?)\b",
    re.IGNORECASE)


def check(name: str):
    """Returns (errors, warnings) — errors are hard fails, warnings are keyword flags."""
    pack = json.load(open(EL / "contract_packs" / f"{name}.json"))
    path = EL / "contracts" / f"{name}.json"
    if not path.exists():
        return ["MISSING contract file"], []
    try:
        c = json.load(open(path))
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"], []
    errs, warns = [], []
    for k in ("construct_definition", "cf_probes", "discrimination_checks", "boundary_notes"):
        if k not in c:
            errs.append(f"missing key {k}")
    if errs:
        return errs, warns
    if c["construct_definition"] != pack["criterion_description"]:
        errs.append("definition NOT verbatim")
    probes = c["cf_probes"]
    if not 4 <= len(probes) <= 6:
        errs.append(f"{len(probes)} probes (need 4-6)")
    for i, p in enumerate(probes):
        pos, neg = (p.get("text_pos") or "").strip(), (p.get("text_neg") or "").strip()
        if not pos or not neg:
            errs.append(f"probe {i}: empty side")
        elif pos == neg:
            errs.append(f"probe {i}: pos == neg")
        hit = FORBIDDEN.search(" ".join([pos, neg, p.get("why") or ""]))
        if hit:
            warns.append(f"probe {i}: FLAG keyword '{hit.group(0)}'")
    if c["discrimination_checks"] != {"min_std": 0.05, "max_frac_at_mode": 0.85}:
        errs.append("discrimination_checks wrong")
    return errs, warns


def main():
    names = sorted(p.stem for p in (EL / "contract_packs").glob("*.json"))
    adj_path = EL / "contract_flag_adjudications.json"
    adjudications = json.load(open(adj_path)) if adj_path.exists() else {}
    n_pass = 0
    report = {}
    for name in names:
        errs, warns = check(name)
        adj = adjudications.get(name, {})
        unadjudicated = [w for w in warns if w.split(":")[0] not in adj]
        ok = not errs and not unadjudicated
        report[name] = {"errors": errs, "warnings": warns,
                        "adjudicated": {k: adj[k] for k in adj
                                        if any(w.startswith(k) for w in warns)},
                        "pass": ok}
        if ok:
            n_pass += 1
        else:
            print(f"{name}: errors={errs} unadjudicated_flags={unadjudicated}")
    json.dump(report, open(EL / "contracts_validation.json", "w"), indent=1)
    print(f"\n{n_pass}/{len(names)} PASS -> {EL / 'contracts_validation.json'}")
    raise SystemExit(0 if n_pass == len(names) else 1)


if __name__ == "__main__":
    main()

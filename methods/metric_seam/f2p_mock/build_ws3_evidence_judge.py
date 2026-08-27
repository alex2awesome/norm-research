"""WS3: evidence-aware judge target M̄(x,Z) prompts for patents_pa (runbook 2026-07-10).

The R7.1 op-marginal null was a level-matching fact: I(M̄(X);Z|X)=0 — an evidence op cannot be
credited against a doc-only judge. This builds the evidence-aware target: the SAME two-pass
judge templates (T1/T2 from build_patents_pa), but the text block carries the application doc
PLUS the serialized PriorArtOps payload (label fields already stripped by ops_pa contract).

Arms (pre-registered):
  evidence  doc + prior-art search record        -> M̄(x,Z), 2 passes (reliability on the NEW
                                                    target — new target = new ceiling)
  filler    doc + length-matched inert text      -> instruction-load control (BEST-PRACTICES
                                                    dossier-collapse guard; Gemma-31B expected
                                                    fine, control is cheap)
Doc-only M̄(x) = the EXISTING tasks/patents_pa/results.jsonl (do not re-run).
Aspects: the 4 R7.1 evidence-dominant hybrids (a26, a34, a60, a35).
4 aspects x 250 items x 2 passes x 2 arms = 4,000 prompts.

Run (sk3, one-GPU pass, gpu_waiter pattern — never contend with OSL lanes):
  python gemma_score_v1.py --prompts tasks/patents_pa/ws3_evidence_prompts.jsonl \
         --out tasks/patents_pa/ws3_evidence_results.jsonl
Then eval: op-marginal of PriorArtOps hybrids vs M̄(x,Z) (expect >0 for evidence-dominant),
vs M̄(x) (expect ~0, the null replicates by design); NullOps twin unchanged.
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("bpa", HERE / "build_patents_pa.py")
bpa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpa)

# bpa hardcodes sk3 paths (it ran there); resolve against THIS repo checkout instead
R = pathlib.Path(__file__).resolve().parents[3]
OUT = R / "outputs/metric_seam_pilot/tasks/patents_pa"
ASPECTS = ["a26", "a34", "a60", "a35"]
PAY_CAP = 2200  # chars of serialized evidence per item

FILLER_SENT = ("This paragraph is neutral placeholder text with no information about this "
               "application; it only matches the length of an attached record. ")


def serialize_payload(pay: dict) -> str:
    """Compact, capped serialization: summary stats + top-2 claims x top-2 refs."""
    if not pay:
        return "No prior-art search record is available for this application."
    head = {k: pay.get(k) for k in ("n_claims", "frac_claims_any_disclose",
                                    "mean_frac_disclose", "max_frac_disclose",
                                    "retrieval_top_scores")}
    lines = [json.dumps(head)]
    for c in (pay.get("claims") or [])[:2]:
        lines.append(f"claim {c.get('claim_num')}: {c.get('element_head','')[:200]} "
                     f"(refs {c.get('n_refs')}, disclosing {c.get('n_disclose')})")
        for ref in (c.get("refs") or [])[:2]:
            lines.append(f"  ref {ref.get('doc_id')}: discloses={ref.get('discloses')}; "
                         f"{(ref.get('vreason') or '')[:150]}")
    return "\n".join(lines)[:PAY_CAP]


def main():
    items = json.load(open(OUT / "items.json"))
    feats = json.load(open(OUT / "pa_features.json"))
    aspects = {x["aspect_id"]: x for x in
               json.load(open(R / "runs/validity_full/v2/patents/aspects.json"))}
    role, doctype = bpa.ROLE
    n = 0
    with open(OUT / "ws3_evidence_prompts.jsonl", "w") as f:
        for it in items:
            dpid = it["datapoint_id"]
            ev = serialize_payload(feats.get(dpid))
            filler = (FILLER_SENT * (len(ev) // len(FILLER_SENT) + 1))[:len(ev)]
            for arm, block in (("evidence", ev), ("filler", filler)):
                text = (f"{it['ctext']}\n\n=== EXAMINER PRIOR-ART SEARCH RECORD "
                        f"(supplementary evidence) ===\n{block}")
                for ch, T in (("pass1", bpa.T1), ("pass2", bpa.T2)):
                    for aid in ASPECTS:
                        asp = aspects[aid]
                        f.write(json.dumps({
                            "channel": f"{arm}_{ch}", "aspect_id": aid, "datapoint_id": dpid,
                            "prompt": T.format(role=role, doctype=doctype, name=asp["name"],
                                               description=asp["description"], text=text),
                        }) + "\n")
                        n += 1
    print(f"{n} prompts -> {OUT / 'ws3_evidence_prompts.jsonl'}")


if __name__ == "__main__":
    main()

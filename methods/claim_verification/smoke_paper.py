#!/usr/bin/env python3
"""Pure-logic smoke for the paper adapter (no GPU/LLM). Run from repo root:
  python -m methods.claim_verification.smoke_paper"""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from claim_verification.paper_adapter import (
    subtractive_body, paragraphs, select_passages, perturb_numbers, stable_pos,
    parse_claims, parse_verify, parse_prior_art,
    support_metrics, retrieval_metrics, prior_art_metrics)

ok = 0
def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    ok += cond

# subtractive body: excludes abstract/references/related-work, keeps method/results
secs = json.dumps({"Preamble": "x" * 100, "Abstract": "ABS " * 200, "Introduction": "intro " * 300,
                   "Related Work": "prior stuff " * 200, "Method": "our method " * 300,
                   "Results": "we achieve 87.2 accuracy " * 100, "References": "refs " * 500})
body, src = subtractive_body(secs, None)
check("subtractive: src=sections", src == "sections")
check("subtractive: excludes refs/relwork/abstract",
      "refs" not in body and "prior stuff" not in body and "ABS" not in body)
check("subtractive: keeps results", "87.2" in body)
body2, src2 = subtractive_body(None, "junkhead " * 200 + "real content sentence. " * 500 +
                               "\nReferences\n" + "ref " * 300)
check("fallback: fulltext used", src2 == "fulltext_fallback" and "ref ref" not in body2)
body3, src3 = subtractive_body(None, "tiny")
check("degenerate -> none", src3 == "none" and body3 == "")

# paragraphs: blank-line split + windowed fallback for PDF-ish text
ps = paragraphs("\n\n".join("This is a substantive paragraph about topic %d. " % i * 6 for i in range(20)))
check("paragraphs: blank-line split", len(ps) == 20)
ps2 = paragraphs("One long PDF blob without blank lines. " * 300)
check("paragraphs: windowed fallback", 10 < len(ps2) < 100 and all(len(p) <= 700 for p in ps2))

# passage selection: retrieves the on-topic paragraph
paras = ["The weather in Paris is mild.", "Our model achieves 87.2 accuracy on ImageNet validation.",
         "We thank the reviewers."] + ["Filler paragraph about unrelated matters number %d." % i for i in range(20)]
sel = select_passages("the model achieves 87.2 accuracy on ImageNet", paras, k=3)
check("select_passages: on-topic top-1", "87.2" in sel[0])

# perturbation: changes every number, deterministic, None when numberless
p = perturb_numbers("improves accuracy by 12.3% over 5 baselines")
check("perturb: changes numbers", p is not None and "12.3" not in p and "5 baselines" not in p, repr(p))
check("perturb: deterministic", p == perturb_numbers("improves accuracy by 12.3% over 5 baselines"))
check("perturb: numberless -> None", perturb_numbers("a novel method for parsing") is None)

# parsers: happy path + garbage + grounding demotion
cl = parse_claims('{"claims": [{"claim": "The method improves BLEU by 2.1 points on WMT14.", "kind": "quantity"}, {"claim": "short", "kind": "x"}]}')
check("parse_claims: keeps valid, drops short", len(cl) == 1 and cl[0]["kind"] == "quantity")
check("parse_claims: garbage -> []", parse_claims("no json here") == [])
v = parse_verify('{"verdict": "FULL", "passage_idx": 0, "span": "achieves 87.2 accuracy", "evidence_type": "data_statistic", "reason": "r"}',
                 ["Our model achieves 87.2 accuracy on ImageNet."])
check("parse_verify: grounded FULL", v["verdict"] == "FULL" and v["grounded"])
v2 = parse_verify('{"verdict": "FULL", "span": "this span is invented"}', ["Nothing like that here."])
check("parse_verify: ungrounded FULL demoted", v2["verdict"] == "PARTIAL" and v2["ungrounded"])
v3 = parse_verify("total garbage", ["x"])
check("parse_verify: garbage -> NONE", v3["verdict"] == "NONE" and not v3["parsed"])
pa = parse_prior_art('{"verdicts": ["ANTICIPATES", "DISTINCT", "bogus"], "best_idx": 0, "reason": "r"}', 4)
check("parse_prior_art: pads + sanitizes", pa["verdicts"] == ["ANTICIPATES", "DISTINCT", "DISTINCT", "DISTINCT"])

# aggregation: planted indices excluded from real novelty readout
rows = [{"verdicts": ["DISTINCT", "ANTICIPATES", "DISTINCT", "DISTINCT"], "self_idx": 1, "foreign_idx": 3},
        {"verdicts": ["PARTIAL", "ANTICIPATES", "DISTINCT", "DISTINCT"], "self_idx": 1, "foreign_idx": 3}]
m = prior_art_metrics(rows)
check("pa metrics: self excluded from real rate", m["pa_anticipated_rate"] == 0.0 and m["pa_partial_rate"] == 0.5)
check("pa metrics: controls read planted slots", m["pa_self_detect"] == 1.0 and m["pa_foreign_distinct"] == 1.0)
sm = support_metrics([{"verdict": "FULL", "grounded": True}, {"verdict": "NONE", "grounded": False}])
check("support metrics", sm["s_support_rate"] == 0.5 and sm["s_grounded_rate"] == 0.5)
rm = retrieval_metrics(["the model achieves 87.2 accuracy"], paras)
check("retrieval metrics finite", rm["r_top1_overlap"] > 0.5)
check("stable_pos deterministic + in-range", stable_pos("abc", 8) == stable_pos("abc", 8) < 8)

print(f"\n{ok} checks passed")
print("SMOKE_PAPER_DONE" if ok >= 22 else "SMOKE_PAPER_FAILED")

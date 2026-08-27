#!/usr/bin/env python3
"""ADDENDUM G1, nc_responded — assemble g1_forms_nc.json from g1_top24_nc.json.
form_a_frozen = original definition verbatim; form_b_paraphrase = hand-authored
meaning-preserving rewording (authored blind to any scores); form_c_minimal =
criterion name only. Frame text matches the peer/cw form template."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PARA = {
 "Concision and focus":
  "Write briefly and stay on the rule itself, so agency staff can easily pull "
  "out and respond to each point without wading through digressions.",
 "r4:P11:Traceable Evidence Use":
  "Judge whether a reader could actually find and verify the support behind the "
  "comment's factual assertions. Top scores name particular studies, datasets, "
  "statutes, official analyses, or clearly described first-hand observations and "
  "say why they matter and where they fall short. Middle scores cite real sources "
  "but never connect them or note their limits. Bottom scores leave claims bare or "
  "lean on vague gestures like 'research shows'.",
 "r1:P09:Substance survives deletion of the position statement":
  "Mentally strike every sentence expressing support, opposition, thanks, emotion, "
  "or a purely procedural ask, and rate what is left over. Top scores retain a "
  "self-standing body of reasons, facts, or requested changes. Middle scores keep "
  "only a sliver, such as a single reason. Bottom scores keep nothing, because the "
  "whole comment was a stance, a sentiment, a +1 to someone else's filing, or a "
  "scheduling request.",
 "r1:P16:Temporal and cumulative effects":
  "Rate whether the comment reasons beyond the immediate moment. Top scores work "
  "through delayed, recurring, compounding, or long-run consequences and describe "
  "what happens over that horizon. Bottom scores stop at a short-term claim about "
  "instant impact. Rate the time horizon the text actually reasons over, not "
  "whether its analysis is right.",
 "r4:P04:Reasoning-based persuasion vs rhetorical amplification":
  "Judge where the comment's persuasive power comes from. Top scores persuade "
  "through worked-out argument and evidence in a level tone, however strongly the "
  "view is held. Middle scores blend genuine argument with heated language. Bottom "
  "scores replace argument entirely with all-caps demands, repetition, insults, or "
  "hyperbole.",
 "r1:P06:Differentiated identification of affected stakeholder groups":
  "Rate whether the comment separates out who is affected and how differently. Top "
  "scores name two or more distinct populations or entities and explain how the "
  "rule hits each one differently. Bottom scores speak only of one undifferentiated "
  "mass like 'the public', or never say who is affected. Rate only the "
  "differentiation stated on the page.",
 "Comment structure, organization, and summaries":
  "Open by saying who you are and why it matters, summarize the key points up "
  "front, organize the body into clearly headed or bulleted sections that flow "
  "logically, and end with a short recap.",
 "Evidence‑supported, verifiable claims":
  "State assertions specifically enough to be checked, and back them with credible "
  "evidence and explicit source citations, ideally independent or scientific ones "
  "where the subject calls for it.",
 "r3:P06:Quantified supporting evidence":
  "Judge whether the position rests on numbers. High = the comment brings relevant "
  "figures, calculations, comparisons, or rates that bear on its argument and are "
  "concrete enough to verify. Low = no checkable quantitative backing anywhere.",
 "Actionable alternatives and drafting in public comments":
  "Offer concrete, workable alternatives that stay within the rule's scope — or "
  "even draft regulatory text ready for adoption, mitigations included — phrased as "
  "clear asks the agency's staff could act on directly.",
 "Provide concrete, decision‑relevant data and examples":
  "Bring ground-level data, specific examples, and operational evidence attached "
  "to each claim, detailed enough that agency analysts can actually work with it.",
 "Commenter identity, stake, and qualifications":
  "Say plainly who is writing, on whose behalf, what their stake in the rule is, "
  "and what expertise or lived experience they bring; where relevant, indicate how "
  "many people or which constituencies stand behind the comment.",
 "Personalization and substance in mass‑comment campaigns":
  "Steer clear of duplicate form letters and obvious templating; the comment "
  "should read as a unique, substantive, human-authored submission (transparent, "
  "genuine use of AI assistance is fine).",
 "Evidentiary support and argumentation quality":
  "Lay out claims clearly and in an organized way, ground them in credible "
  "peer-reviewed or empirical sources with citations and concrete examples, keep "
  "the signal-to-noise high, and engage likely counterarguments head-on.",
 "r1:P17:Quantified scale of real-world impact":
  "Judge whether the comment gives specific, argument-relevant numbers that "
  "establish how big the problem or benefit is. Top scores offer figures tied to "
  "this particular matter — dollars, percentages, counts, timeframes — that make "
  "the magnitude visible. Bottom scores give no numbers, or only a generic or "
  "recycled statistic unconnected to the issue. Judge the figures' specificity and "
  "relevance, not which side they help.",
 "r1:P19:Verifiable, attributable evidentiary sourcing":
  "Rate how firmly the comment's factual claims are anchored to identifiable "
  "sources a reader could check. Top scores cite named studies, agency data, "
  "statutes, or official reports that genuinely support the claim. Bottom scores "
  "assert facts with no source, or attribute them to weak or irrelevant material "
  "like unnamed blogs, hearsay, or 'as I understand it'. Rate the sourcing, not "
  "your agreement.",
 "r4:P25:Specific, precise proposed regulatory fix":
  "Judge how precisely the comment specifies the change it wants. Top scores "
  "supply exact substitute language, a definite numeric threshold, or an "
  "unmistakable procedural modification. Middle scores indicate which way the rule "
  "should move but not what the change would say. Bottom scores register only "
  "support or opposition with no proposed alternative.",
 "Evidentiary rigor and sourcing in comments":
  "Ground every argument in specific, credible, well-documented evidence with "
  "clear sourcing; use sound methods, own up to uncertainty, and avoid unsupported "
  "claims or conclusions that outrun the evidence.",
 "Concrete, context‑specific, domain‑aware impact evidence":
  "Show harms or benefits through detailed real-world examples and technical "
  "specifics: pinpoint who and where is affected, and bring sector-specific data "
  "and practitioner knowledge that illuminate implementation consequences.",
 "r2:P08:Distributional incidence and equity":
  "Rate whether the comment sorts out who pays and who gains. High when it names "
  "specific populations, firms, places, or user groups facing distinct costs or "
  "benefits and explains why their exposure or access differs. Low when impacts "
  "are attributed only to a generic public.",
 "r3:P11:Credible alternative meeting the regulatory objective":
  "Judge whether the comment works out a real alternative. High = it proposes a "
  "specific substitute, exception, phase-in, or variant and argues why that route "
  "can still deliver the rule's objective, tradeoffs named. Low = it merely "
  "supports or opposes the proposal as drafted.",
 "r1:P12:Detail-level concreteness of any proposed alternative":
  "Rate how completely any alternative the comment floats is spelled out in the "
  "text itself. Top scores describe the mechanism — steps, parts, numbers, "
  "procedure — fully enough for another reader to implement or assess it. Bottom "
  "scores name an alternative without describing it, defer to a solution kept "
  "outside the comment ('contact me, I have one'), or propose no alternative at "
  "all.",
 "r4:P15:Viable alternative design":
  "Judge whether the comment presents a workable alternative regulatory design "
  "that still serves the proposal's stated goal while curing the flaw it "
  "identifies. Top scores describe the alternative clearly enough that its "
  "operation is evident. Middle scores gesture at a direction without any design. "
  "Bottom scores offer nothing beyond scrapping the proposal or keeping it as is.",
 "r2:P23:Feasible alternative design":
  "Judge whether the comment puts forward an implementable variant. High when it "
  "proposes a concrete alternative, carve-out, threshold, or safeguard and shows "
  "how the variant still fulfills the proposal's legitimate aim. Low when it gives "
  "only support, opposition, or an unelaborated demand. Score in the middle when "
  "an alternative is named but its fit to the objective goes unexplained.",
}

top = json.load(open(HERE / "g1_top24_nc.json"))
assert len(top) == 24
out = []
for r in top:
    nm, d = r["name"], r["definition"]
    short = nm.split(":", 2)[-1]
    p = PARA[nm]
    assert p.strip() and p.strip() != d.strip(), nm
    out.append({
        "name": nm, "col_XAmined": r["col_XAmined"], "auc_dev": r["auc_dev"],
        "form_a_frozen": f"CRITERION: {short}\nINSTRUCTION: {d}\n\nAnswer with one token:",
        "form_b_paraphrase": f"CRITERION: {short}\nINSTRUCTION: {p}\n\nAnswer with one token:",
        "form_c_minimal": f"CRITERION: {short}\n\nAnswer with one token:",
    })
json.dump(out, open(HERE / "g1_forms_nc.json", "w"), indent=1, ensure_ascii=False)
print(f"wrote {len(out)} forms  G1_NC_FORMS_DONE")

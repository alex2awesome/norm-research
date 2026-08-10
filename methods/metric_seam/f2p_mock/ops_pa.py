"""PriorArtOps — the pilot op library + a prior-art disclosure EVIDENCE op (patents R7.1).

The real machinery (other-thread pipeline, datasets/patents/processed/ on sk3): BM25/dense
retrieval pulls K=8 candidate prior-art references per claim, then Gemma reads each
(claim element, reference) pair and emits a disclosure verdict + supporting spans + a one-line
reason (option3_claims_gemma_scale.jsonl, 59,937 claims over 21,447 applications).

Here NOTHING is retrieved or verified at call time: `prior_art(dpid)` looks up the
ALREADY-COMPUTED retrieval + disclosure record for this application. Hybrid programs call it
exactly where they would otherwise need a patent examiner's search. Formally an EVIDENCE op:
Z touches corpus state far beyond x (the prior-art literature + a reading model), so
I(M; X, Z) >= I(M; X) — the channel ceiling itself moves.

Payload per dpid (None if the application has no measurement):
  n_claims                 number of measured claims for this application
  frac_claims_any_disclose fraction of claims with >=1 disclosing reference
  mean_frac_disclose       mean over claims of n_disclose/n_refs
  max_frac_disclose        max over claims of n_disclose/n_refs
  claims: [ { claim_num, element_head (<=300ch), n_refs, n_disclose,
              refs: [ { doc_id, discloses (bool), vreason (<=200ch),
                        span_head (<=300ch, "" if none) } ] } ]
  retrieval_top_scores     top-5 corpus retrieval similarity scores for the application
Ground-truth fields (label, rejection_type, gold_docs, is_gold, gold_disclose, judgement,
rejected_*) are deliberately EXCLUDED — they are the anchors / supervision.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hybrids"))
from ops import Ops


class PriorArtOps(Ops):
    def __init__(self, features_path, corpus_path=None):
        super().__init__(corpus_path=corpus_path)
        self._feat = json.load(open(features_path))

    def prior_art(self, dpid):
        """Prior-art retrieval + disclosure evidence op (see module docstring)."""
        return self._feat.get(dpid)


class NullPriorArtOps(PriorArtOps):
    """Ablation twin: identical interface, evidence op returns nothing —
    isolates the prior-art machinery's marginal contribution."""
    def __init__(self, features_path, corpus_path=None):
        Ops.__init__(self, corpus_path=corpus_path)
        self._feat = {}

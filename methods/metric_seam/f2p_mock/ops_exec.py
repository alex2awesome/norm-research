"""ExecOps — the pilot op library + a MOCKED test-transition evidence op.

The real machinery (datasets/code-review/pr_test_execution/): an LLM/heavy pipeline constructs
an era Docker image and identifies test files (factory/prep_repo.py — the LLM slot of the
channel), then code executes the transplanted tests inside the image
(factory/exec_leg/run_local.sh — the code slot) and emits per-PR verdicts.

Here NOTHING is built or executed: `test_transition(dpid)` looks up the ALREADY-COMPUTED
transplant outcome for this PR (transplant_consolidated_*.parquet). Hybrid programs call it
exactly where they would otherwise have to implement F2P/P2F. Formally an EVIDENCE op:
Z touches world state far beyond the diff text (repo checkout, dependency era, execution),
so I(M; X, Z) >= I(M; X) — the channel ceiling itself moves.

Payload per dpid (None if the PR has no measurement):
  label            transplant_pr_label: pinned | partial_pinned | vacuous | none |
                   indeterminate | error_* (pinned ~ transplanted tests fail on base and
                   pass with patch = F2P-causal-like; vacuous ~ tests pass WITHOUT the patch)
  n_assertion_fail, n_vacuous_pass, n_compile_fail, n_setup_fail, n_uncollected  (counts)
  test_byte_ratio, n_files_total, n_files_test, test_only_ratio,
  n_lines_added, n_lines_deleted, language
Acceptance labels (judgement / days_open) are deliberately EXCLUDED — they are the anchors.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hybrids"))
from ops import Ops


class ExecOps(Ops):
    def __init__(self, features_path, corpus_path=None):
        super().__init__(corpus_path=corpus_path)
        self._feat = json.load(open(features_path))

    def test_transition(self, dpid):
        """MOCKED F2P/P2F evidence op (see module docstring). Returns dict or None."""
        return self._feat.get(dpid)


class NullExecOps(ExecOps):
    """Ablation twin: identical interface, evidence op returns nothing —
    isolates the mocked machinery's marginal contribution."""
    def __init__(self, features_path, corpus_path=None):
        Ops.__init__(self, corpus_path=corpus_path)
        self._feat = {}

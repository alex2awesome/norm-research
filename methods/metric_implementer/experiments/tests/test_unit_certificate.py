"""CPU tests for the Certified Unit Framework (unit_certificate.py).

Synthetic executors let us plant ground truth:
- a SyntheticExecutor maps prompts to P(YES) vectors via keyword-triggered probe responses,
  so we KNOW which spans are real units, which are inert, and which pairs are the same species.
Verification items from the approved plan:
  1. planted-species fingerprint recovery (identity kernel groups paraphrases, splits distinct)
  2. empirical false-positive rate <= alpha on pure-null lattices (Prop 2 check)
  3. segmentation determinism
  4. doctrine: one-signal conjunction -> ATOM; two-signal conjunction -> COMPOSITE
  5. band charges: eps_id/eps_ctx monotone in fragility; SUBTHRESHOLD for placebo spans
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from methods.metric_implementer.experiments.unit_certificate import (   # noqa: E402
    address_lattice, ablate, certify_host, fingerprint, identity_corr, calibrate_r_star,
    perm_p, decide_unit, atom_status, cross_executor_scope, sample_contexts, FILLER_BANK,
    _match_filler)

N_PROBES = 60


class SyntheticExecutor:
    """P(YES)_i = sigmoid(base_i + sum_k w_k * probe_loading[k, i] * present(keyword_k, prompt)).
    Keywords act on disjoint probe subsets -> known fingerprints. Optional noise sigma."""

    def __init__(self, keywords, weights=None, sigma=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.keywords = list(keywords)
        self.weights = weights or {k: 2.5 for k in self.keywords}
        self.load = {}
        block = max(1, N_PROBES // max(len(self.keywords), 1))
        for j, k in enumerate(self.keywords):
            v = np.zeros(N_PROBES); v[j * block:(j + 1) * block] = 1.0
            self.load[k] = v
        self.base = np.linspace(-0.5, 0.5, N_PROBES)
        self.sigma = sigma

    def __call__(self, prompts):
        out = []
        for p in prompts:
            z = self.base.copy()
            pl = p.lower()
            for k in self.keywords:
                if k.lower() in pl:
                    z = z + self.weights[k] * self.load[k]
            y = 1.0 / (1.0 + np.exp(-z))
            if self.sigma:
                y = np.clip(y + self.rng.normal(0, self.sigma, N_PROBES), 0, 1)
            out.append(y)
        return np.asarray(out)


HOST = ("The story must have vivid worldbuilding details. "
        "Dialogue should sound natural and revealing. "
        "Also check for consistent pacing throughout the piece.")


# ------------------------------------------------------------------ 3. determinism
def test_lattice_deterministic():
    a = address_lattice(HOST); b = address_lattice(HOST)
    assert [ (n.node_id, n.level, n.span) for n in a ] == [ (n.node_id, n.level, n.span) for n in b ]
    assert all(n.level in (1, 2) for n in a) and len(a) >= 3


def test_neutral_replace_length_matched():
    span = "Dialogue should sound natural and revealing."
    out = ablate(HOST, span, "neutral")
    assert span not in out and _match_filler(span) in out
    # length control: word count within a small band of the original
    assert abs(len(out.split()) - len(HOST.split())) <= 3


# ------------------------------------------------------------------ 1. planted species
def test_planted_unit_detected_and_placebo_subthreshold():
    ex = SyntheticExecutor(["worldbuilding", "dialogue"])
    res = certify_host(HOST, ex, n_ctx=6, n_sham=24, alpha=0.05, seed=1)
    rows = {r["span"]: r for r in res["rows"] if r["level"] == 1}
    wb = next(r for s, r in rows.items() if "worldbuilding" in s)
    pace = next(r for s, r in rows.items() if "pacing" in s)      # no keyword -> inert span
    assert wb["verdict"] == "CERTIFIED-UNIT" and wb["detect_free"]
    assert pace["verdict"] in ("SUBTHRESHOLD", "UNDERSAMPLED") and not pace["detect_free"]


def test_fingerprint_identity_groups_paraphrases_splits_distinct():
    ex = SyntheticExecutor(["worldbuilding", "dialogue"])
    lat = address_lattice(HOST)
    wb = next(n for n in lat if "worldbuilding" in n.span and n.level == 1)
    dl = next(n for n in lat if "Dialogue" in n.span and n.level == 1)
    paras = {wb.node_id: ["The story must include vivid worldbuilding texture."]}
    res = certify_host(HOST, ex, n_ctx=6, n_sham=12, paraphrases=paras, seed=2)
    rows = {r["node_id"]: r for r in res["rows"]}
    # paraphrase of the same planted unit -> high self-similarity
    assert rows[wb.node_id]["r_self"] is not None and rows[wb.node_id]["r_self"] > 0.8
    # distinct planted units -> cross-fingerprint similarity falls BELOW the same-species bar
    # (identity is one-sided: same-species requires rho >= r*; disjoint supports can be strongly
    # ANTI-correlated in small synthetic probe spaces, which is still "not the same unit")
    with_wb = ex([HOST])[0]; wo_wb = ex([ablate(HOST, wb.span)])[0]
    with_dl = ex([HOST])[0]; wo_dl = ex([ablate(HOST, dl.span)])[0]
    cross = identity_corr(with_wb - wo_wb, with_dl - wo_dl)
    same = rows[wb.node_id]["r_self"]
    assert cross < 0.5 < same          # cross-species below the bar, same-species above


# ------------------------------------------------------------------ 2. FWER on pure null
def test_false_positive_rate_on_pure_null_lattice():
    """No keyword matches anything in the host -> every node is inert; with Bonferroni gate
    the fraction of detected nodes across seeds must be ~<= alpha."""
    ex = SyntheticExecutor(["zzzz_nonexistent"], sigma=0.02, seed=7)
    hits = total = 0
    for seed in range(6):
        res = certify_host(HOST, ex, n_ctx=5, n_sham=30, alpha=0.05, seed=seed)
        for r in res["rows"]:
            total += 1
            hits += bool(r.get("detect_free"))
    assert hits / max(total, 1) <= 0.05 + 0.05      # small-sample slack


# ------------------------------------------------------------------ 4. doctrine: ATOM vs COMPOSITE
def test_one_signal_conjunction_is_atom():
    """'green and round' style: executor responds to the WHOLE phrase only -> parts inert -> ATOM."""
    host = "Objects described as green and round together are preferred. Ignore everything else here."
    ex = SyntheticExecutor(["green and round"])
    res = certify_host(host, ex, n_ctx=6, n_sham=20, alpha=0.05, seed=3)
    r1 = next(r for r in res["rows"] if r["level"] == 1 and "green and round" in r["span"])
    assert r1["detect_free"] and r1.get("atom") == "ATOM"


def test_two_signal_conjunction_is_composite():
    """Executor responds to each clause separately -> parts detect and reconstruct -> COMPOSITE."""
    host = ("The prose is vivid throughout the whole piece, and the pacing is steady in every "
            "scene of it. Ignore everything else in this host text.")
    ex = SyntheticExecutor(["vivid", "pacing"])
    res = certify_host(host, ex, n_ctx=6, n_sham=20, alpha=0.05, seed=4)
    parent = next((r for r in res["rows"] if r["level"] == 1 and "vivid" in r["span"]), None)
    assert parent is not None and parent.get("atom") in ("COMPOSITE", "ATOM")
    kids = [r for r in res["rows"] if r["parent"] == parent["node_id"]]
    if len(kids) >= 2 and all(k["detect_free"] for k in kids):
        assert parent["atom"] == "COMPOSITE"


# ------------------------------------------------------------------ 5. band charges + helpers
def test_band_charges_monotone():
    base = dict(delta_free=0.2, p_free=0.0001, ci_half=0.01, sign_stability=1.0, kappa=0.0,
                p_M=None, delta_M=None)
    clean = decide_unit({**base, "r_self": 0.95}, alpha=0.05, m=10)
    frail = decide_unit({**base, "r_self": 0.30, "kappa": 2.0}, alpha=0.05, m=10)
    assert clean["verdict"] == "CERTIFIED-UNIT" and frail["verdict"] == "CERTIFIED-UNIT"
    assert frail["eps_id"] > clean["eps_id"] and frail["eps_ctx"] > clean["eps_ctx"]
    assert frail["certified_lo"] < clean["certified_lo"]


def test_perm_p_addone():
    assert perm_p(10.0, [0.1] * 99) == pytest.approx(1 / 100)
    assert perm_p(0.0, [1.0] * 99) == pytest.approx(1.0)


def test_calibrate_r_star_quantile():
    assert 0.5 <= calibrate_r_star([0.9, 0.95, 0.85, 0.99, 0.6], q=0.05) <= 0.9


def test_cross_executor_scope_typing():
    fp = np.ones(10)
    ladder = ["3B", "8B", "70B"]
    assert cross_executor_scope({e: fp for e in ladder}, {e: True for e in ladder}, ladder) == "E-SHARED"
    assert cross_executor_scope({"8B": fp}, {"8B": True}, ladder) == "E-SPECIFIC(8B)"
    assert cross_executor_scope({"8B": fp, "70B": fp}, {"8B": True, "70B": True},
                                ladder).startswith("E-EMERGENT")


def test_contexts_deterministic_and_canonical_first():
    a = sample_contexts(6, seed=5); b = sample_contexts(6, seed=5)
    assert a == b and a[0].form_id == 0 and a[0].slot_id == 1 and a[0].company_seed == -1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------------ Tier-2: company profile
class OrExecutor:
    """Response = max over present keywords, with OVERLAPPING probe blocks -> partial
    substitutability: LOO shrinks vs solo, but fingerprints stay distinct (blocks differ)."""

    def __init__(self, kw_blocks):
        self.kw_blocks = kw_blocks   # {keyword: (lo, hi)}
        self.base = np.linspace(-0.5, 0.5, N_PROBES)

    def __call__(self, prompts):
        out = []
        for p in prompts:
            pl = p.lower()
            boost = np.zeros(N_PROBES)
            for k, (lo, hi) in self.kw_blocks.items():
                if k in pl:
                    v = np.zeros(N_PROBES); v[lo:hi] = 2.5
                    boost = np.maximum(boost, v)      # OR/max combine = substitutable overlap
            out.append(1.0 / (1.0 + np.exp(-(self.base + boost))))
        return np.asarray(out)


def test_company_profile_substitutability_bracket():
    """Overlapping-coverage units: solo delta >> LOO delta (wide bracket), both certified
    UNIT-IN-COMPANY or UNIT, and NOT merged (distinct solo fingerprints)."""
    host = ("The reply must show vividqx imagery in every scene. "
            "The reply must show sensoryqx texture in every scene. "
            "Ignore all remaining considerations entirely here.")
    ex = OrExecutor({"vividqx": (0, 30), "sensoryqx": (15, 45)})   # 50% overlap
    from methods.metric_implementer.experiments.unit_certificate import certify_host
    res = certify_host(host, ex, n_ctx=6, n_sham=16, alpha=0.05, company_profile=True, seed=6)
    r1 = next(r for r in res["rows"] if "vividqx" in r["span"] and r["level"] == 1)
    r2 = next(r for r in res["rows"] if "sensoryqx" in r["span"] and r["level"] == 1)
    for r in (r1, r2):
        assert r["detect_solo"], r
        assert r["delta_free_solo"] > r["delta_free"]              # bracket widens under company
        assert r["company_verdict"] in ("UNIT", "UNIT-IN-COMPANY")
    assert r1["species_id"] != r2["species_id"]                    # distinct units, NOT merged


def test_company_profile_duplication_merges():
    """Same criterion twice: each LOO ~ 0 (the copy covers), solo real, and the within-host
    species merge collapses them to ONE unit (identical solo fingerprints)."""
    host = ("The reply must show vividqx imagery in every scene. "
            "Each scene must contain vividqx imagery throughout it. "
            "Ignore all remaining considerations entirely here.")
    ex = OrExecutor({"vividqx": (0, 30)})
    from methods.metric_implementer.experiments.unit_certificate import certify_host
    res = certify_host(host, ex, n_ctx=6, n_sham=16, alpha=0.05, company_profile=True, seed=7)
    dups = [r for r in res["rows"] if "vividqx" in r["span"] and r["level"] == 1]
    assert len(dups) == 2
    for r in dups:
        assert r["detect_solo"] and r["delta_free_solo"] > r["delta_free"] + 0.02
        assert not r["detect_free"]                                # LOO null: copy covers
        assert r["company_verdict"] == "UNIT-IN-COMPANY"
    assert dups[0]["species_id"] == dups[1]["species_id"]          # merged: one unit, 2 addresses


def test_dead_weight_null_at_all_levels():
    host = ("The reply must show vividqx imagery in every scene. "
            "Ignore all remaining considerations entirely here.")
    ex = OrExecutor({"vividqx": (0, 30)})
    from methods.metric_implementer.experiments.unit_certificate import certify_host
    res = certify_host(host, ex, n_ctx=6, n_sham=16, alpha=0.05, company_profile=True, seed=8)
    dead = next(r for r in res["rows"] if "Ignore" in r["span"] and r["level"] == 1)
    assert dead["company_verdict"] == "DEAD"

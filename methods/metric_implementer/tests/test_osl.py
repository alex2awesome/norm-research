"""CPU tests for the OSL executor-scaling pipeline (battery / sweep primitives / fits)."""
import json
import os

import numpy as np
import pytest

from ..experiments import osl_battery as ob
from ..experiments import osl_sweep as osw
from ..experiments import osl_fit as of

TEXTS = [
    "Why did the chicken cross the road? Nobody knows, but it had a plan and a dream to follow.",
    "A man walks into a bar with 3 ducks and orders a round of drinks for everyone in the room.",
    "The committee met on Tuesday to discuss the annual budget shortfall in painful detail again.",
    "She told him the punchline twice, and twice he stared blankly at the wall behind her head.",
    "Knock knock. Who is there? An owl. An owl who? Exactly, said the owl, flying away quickly.",
    "My therapist says I have a preoccupation with vengeance. We will see about that, I replied.",
    "I have 99 problems and unsolicited advice about all of them from my extended family members.",
    "The weather report promised sunshine, so naturally it rained for eleven consecutive days.",
] * 6


class TestBattery:
    def test_truth_correct_and_deterministic(self):
        items = ob.build_battery(TEXTS, seed=0, target=80)
        items2 = ob.build_battery(TEXTS, seed=0, target=80)
        assert json.dumps(items, sort_keys=True) == json.dumps(items2, sort_keys=True)
        for it in items:  # recompute a verifiable subset of truths by independent code
            t, c = it["text"], it["criterion"]
            if "question mark" in c and "AND" not in c:
                assert it["truth"] == ("?" in t)
            if c.startswith("This text is longer than") and "AND" not in c:
                k = int(c.split("longer than ")[1].split(" ")[0])
                assert it["truth"] == (len(t.split()) > k)

    def test_balance_and_families(self):
        items = ob.build_battery(TEXTS, seed=0, target=80)
        truths = [it["truth"] for it in items]
        assert 0.4 <= np.mean(truths) <= 0.6
        assert {"threshold", "presence"} <= {it["family"] for it in items}

    def test_auc(self):
        y = np.array([0] * 50 + [1] * 50)
        s = y + np.random.default_rng(0).normal(0, 0.3, 100)
        assert ob._auc(s, y) > 0.9
        assert abs(ob._auc(np.random.default_rng(1).normal(0, 1, 100), y) - 0.5) < 0.15


class TestRecovery:
    def test_planted_signal_recovered(self):
        rng = np.random.default_rng(0)
        S = rng.normal(0, 1, (20, 200))
        y = 0.7 * S[3] + 0.5 * S[7] + rng.normal(0, 0.3, 200)
        out = osw.greedy_recovery(S, y)
        assert out["r_odd"] > 0.6
        assert 3 in out["selected"] or 7 in out["selected"]

    def test_null_signal_low(self):
        rng = np.random.default_rng(1)
        out = osw.greedy_recovery(rng.normal(0, 1, (20, 200)), rng.normal(0, 1, 200))
        assert out["r_odd"] < 0.35

    def test_form_split_reliability(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, 300)
        per_form = np.vstack([base + rng.normal(0, 0.4, 300) for _ in range(4)])
        rel = osw.form_split_reliability(per_form, np.arange(1, 300, 2))
        assert 0.6 < rel <= 1.0
        # pure-noise forms -> reliability near 0, not near 1
        pf_null = rng.normal(0, 1, (4, 300))
        rel0 = osw.form_split_reliability(pf_null, np.arange(1, 300, 2))
        assert not np.isfinite(rel0) or abs(rel0) < 0.4

    def test_stable_pick(self):
        xs = [f"criterion {i}" for i in range(100)]
        assert osw._stable_pick(xs, 10) == osw._stable_pick(list(reversed(xs)), 10)


class TestFits:
    def test_logistic_recovers_planted(self):
        z = np.linspace(-2, 4, 14)
        hits = 0
        for seed in range(10):  # CI coverage over realizations, not one draw
            y = of._logistic(z, 0.8, 1.5, 0.5) + np.random.default_rng(seed).normal(0, 0.02, len(z))
            f = of.fit_logistic(z, y)
            assert abs(f["L"] - 0.8) < 0.1
            ci = of.profile_ci_L(z, y, f)
            hits += int(ci[0] - 1e-9 <= 0.8 <= ci[1] + 1e-9)
        assert hits >= 8

    def test_saturating_beats_linear_when_saturated(self):
        rng = np.random.default_rng(1)
        z = np.linspace(-2, 5, 14)
        y = of._logistic(z, 0.7, 2.0, 0.0) + rng.normal(0, 0.02, len(z))
        lg, ln = of.fit_logistic(z, y), of.fit_linear(z, y)
        assert of._aicc(lg["rss"], len(z), 3) < of._aicc(ln["rss"], len(z), 2)

    def test_loeo_positive_on_real_signal(self):
        rng = np.random.default_rng(2)
        z = np.linspace(-2, 4, 12)
        y = of._logistic(z, 0.75, 1.2, 0.8) + rng.normal(0, 0.03, len(z))
        assert of.loeo(z, y, "logistic")["r2_loeo"] > 0.5

    def test_pca_pls_sanity(self):
        rng = np.random.default_rng(3)
        f = rng.normal(0, 1, 12)
        M = np.outer(f, rng.normal(0, 1, 6)) + rng.normal(0, 0.1, (12, 6))
        pc, evr = of.pca1(M)
        assert abs(of.spearman(pc, f)) > 0.9 and evr > 0.7
        y = 0.6 * f + rng.normal(0, 0.2, 12)
        assert of.pls_loeo(M, y) > 0.2

    def test_family_permutation(self):
        rng = np.random.default_rng(4)
        fams = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
        null = of.family_permutation(rng.normal(0, 1, 15), fams, n_perm=500)
        assert null["p"] > 0.01  # no family structure -> not significant
        struct = of.family_permutation(
            np.r_[rng.normal(0, .1, 5), rng.normal(1, .1, 5), rng.normal(2, .1, 5)],
            fams, n_perm=500)
        assert struct["p"] < 0.05

    def test_planted_reference_selects_by_capability_not_outcome(self):
        curves = {
            "plant": {"z": [0.0, 1.0, 2.0, 3.0], "y": [0.99, 0.10, 0.20, 0.30],
                      "execs": ["low", "mid", "high", "frontier"]}
        }
        ref = of.planted_capability_reference(curves, ["plant"], top_k=2)
        assert ref["value"] == pytest.approx(0.25)  # y at z=2,3; not the top-y values 0.99,0.30
        assert ref["members"]["plant"] == ["high", "frontier"]
        assert ref["is_proved_ceiling"] is False


class TestConsensus:
    def _mk(self, noise_by_exec, n_metrics=6, n=200, seed=0):
        """Executors = noisy readers of a shared latent per metric; noise ~ inverse capability."""
        rng = np.random.default_rng(seed)
        latent = rng.normal(0, 1, (n_metrics, n))
        mbars = {}
        fams = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"}
        for e, sd in noise_by_exec.items():
            mb = latent + rng.normal(0, sd, (n_metrics, n))
            pf = np.stack([mb + rng.normal(0, 0.1, mb.shape) for _ in range(4)], axis=1)
            mbars[e] = {"family": fams[e], "names": [f"m{i}" for i in range(n_metrics)],
                        "kinds": ["bank"] * n_metrics, "m_bar": mb, "per_form": pf}
        return mbars

    def test_agreement_ranks_by_noise(self):
        noise = {"a1": 0.2, "a2": 0.5, "b1": 1.0, "b2": 2.0, "c1": 4.0}
        from ..experiments.osl_fit import consensus_agreement
        ca = consensus_agreement(self._mk(noise), top_k=3,
                                 battery_z={"a1": 3, "a2": 2, "b1": 1, "b2": 0, "c1": -1})
        means = {e: np.nanmean(ca["agree"][e]) for e in noise}
        order = sorted(means, key=lambda e: -means[e])
        assert order == ["a1", "a2", "b1", "b2", "c1"]  # agreement tracks reader fidelity
        assert np.nanmean(ca["frontier_agreement"]) > 0.5  # low-noise frontier agrees

    def test_frontier_floor_detects_underdetermination(self):
        # all executors read a SHARED latent for metric 0 but IDIOSYNCRATIC latents for metric 1
        rng = np.random.default_rng(1)
        n = 200
        shared = rng.normal(0, 1, n)
        mbars = {}
        for k, e in enumerate(["a1", "a2", "b1", "b2"]):
            own = rng.normal(0, 1, n)
            mb = np.vstack([shared + rng.normal(0, 0.3, n), own])
            pf = np.stack([mb + rng.normal(0, 0.1, mb.shape) for _ in range(4)], axis=1)
            mbars[e] = {"family": "A" if k < 2 else "B", "names": ["shared", "own"],
                        "kinds": ["bank", "bank"], "m_bar": mb, "per_form": pf}
        from ..experiments.osl_fit import consensus_agreement
        ca = consensus_agreement(mbars, top_k=4,
                                 battery_z={"a1": 1, "a2": 1, "b1": 1, "b2": 1})
        assert ca["frontier_agreement"][0] > 0.7   # determinate criterion: frontier converges
        assert abs(ca["frontier_agreement"][1]) < 0.3  # underdetermined: frontier disagrees


class TestSweepFake:
    def test_fake_end_to_end(self, tmp_path):
        """Whole sweep path on FakeVLLM: freeze from synthetic npz -> run -> valid output JSON."""
        src = tmp_path / "src"
        src.mkdir()
        rng = np.random.default_rng(0)
        for i in range(6):
            np.savez(src / f"m{i}_sigs.npz",
                     name=f"Metric number {i} with a name",
                     prompts=np.array([f"criterion {i}-{j} about jokes" for j in range(15)],
                                      dtype=object),
                     sigs=rng.random((15, 40)), M_i=rng.random(40))
        bank = {"merged_groups": [{"merged_name": f"Metric number {i} with a name",
                                   "merged_description": "a long description of what this metric "
                                   "measures in humorous texts overall"} for i in range(6)]}
        bank_p = tmp_path / "bank.json"
        bank_p.write_text(json.dumps(bank))
        frz_p = tmp_path / "freeze.json"
        # monkeypatch _load_texts to avoid manifest dependency
        import methods.metric_implementer.experiments.osl_sweep as sweep_mod
        orig = sweep_mod._load_texts
        sweep_mod._load_texts = lambda task, n, cfg: (TEXTS * 10, None)
        try:
            osw.main(["--build-freeze", "--src-dir", str(src), "--bank-r2", str(bank_p),
                      "--n-metrics", "4", "--k-criteria", "8", "--n-probes", "20",
                      "--out", str(frz_p)])
            items = ob.build_battery(TEXTS, seed=0, target=40)
            bat_p = tmp_path / "battery.json"
            bat_p.write_text(json.dumps({"meta": {}, "items": items}))
            out_p = tmp_path / "llama1b.json"
            osw.main(["--executor", "llama1b", "--freeze", str(frz_p),
                      "--battery", str(bat_p), "--fake", "--out", str(out_p)])
            res = json.load(open(out_p))
            assert res["final"] and len(res["metrics"]) == 9  # 4 bank + 5 planted
            assert np.isfinite(res["battery"]["auc"])
            assert all(("r_odd" in m) or m["excluded"] for m in res["metrics"])
            # pairwise pilot arm on the same freeze (FakeVLLM)
            pw_p = tmp_path / "pair_llama1b.npz"
            osw.main(["--pairwise", "--executor", "llama1b", "--freeze", str(frz_p),
                      "--n-pairs", "12", "--fake", "--out", str(pw_p)])
            z = np.load(pw_p, allow_pickle=True)
            assert z["pref_ab"].shape == (9, 12) and z["pref_ba"].shape == (9, 12)
            assert np.isfinite(z["pref_ab"]).mean() > 0.9
            # mbar canonical-only diagnostic mode + probe-window override
            mb_p = tmp_path / "mbar1_llama1b.npz"
            osw.main(["--mbar-only", "--executor", "llama1b", "--freeze", str(frz_p),
                      "--n-forms", "1", "--probe-window", "5:25", "--fake",
                      "--out", str(mb_p)])
            z2 = np.load(mb_p, allow_pickle=True)
            assert z2["m_bar"].shape == (9, 20)
        finally:
            sweep_mod._load_texts = orig

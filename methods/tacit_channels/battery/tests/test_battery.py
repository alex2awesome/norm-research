"""Battery harness tests: catalog coverage (zero orphans), W0 probe functions on synthetic
fixtures, convergence assembly."""
import json
import re

import numpy as np
import pytest

from methods.tacit_channels.battery.artifacts import ArtifactContext, item_agreement
from methods.tacit_channels.battery.probes.generalization import typicality_decay_stats
from methods.tacit_channels.battery.probes.scaling import classify_scaling
from methods.tacit_channels.battery.probes.statability import zero_corr_stats
from methods.tacit_channels.battery.registry import all_probes, covered_catalog_ids

CATALOG = "notes/2026-07-22__tacit-knowledge-operationalization-catalog.md"


def _catalog_entry_ids() -> set:
    """Entry numbers 1..42 map to A1-14 / B15-28 / C29-42."""
    text = open(CATALOG).read()
    nums = {int(m) for m in re.findall(r"^(\d+)\. \*\*", text, re.MULTILINE)}
    ids = set()
    for n in nums:
        ids.add(("A" if n <= 14 else "B" if n <= 28 else "C") + str(n))
    return ids


def test_registry_covers_entire_catalog():
    expected = _catalog_entry_ids()
    assert len(expected) >= 40, f"catalog parse found only {len(expected)} entries"
    covered = covered_catalog_ids()
    orphans = expected - covered
    assert not orphans, f"catalog entries not covered by any probe/gate/framing: {sorted(orphans)}"


def test_all_probes_have_falsifiers_and_waves():
    specs = all_probes()
    assert len(specs) >= 25
    for s in specs:
        assert s.falsifier, s.id
        assert s.wave in (0, 1, 2, 3, 4), s.id
        if s.wave == 0:
            assert s.compute is not None, f"{s.id} is wave 0 but not runnable"


def test_zero_corr_stats_dissociation():
    rng = np.random.default_rng(0)
    n = 300
    agreement = np.clip(rng.normal(0.75, 0.1, n), 0, 1)     # above chance
    conf_indep = rng.uniform(0, 0.5, n)                      # unrelated confidence
    s = zero_corr_stats(conf_indep, agreement)
    assert s["mean_item_agreement"] > 0.6
    assert abs(s["conf_acc_corr"]) < 0.15                    # zero-correlation signature
    assert s["guess_quartile_agreement"] > 0.6               # knowledge on "guess" trials
    # explicit case: confidence tracks accuracy
    conf_tracking = agreement + rng.normal(0, 0.02, n)
    s2 = zero_corr_stats(conf_tracking, agreement)
    assert s2["conf_acc_corr"] > 0.8


def test_typicality_decay_direction():
    rng = np.random.default_rng(1)
    typ = rng.uniform(0, 1, 400)
    # agreement decays off the typical core
    agr = 0.5 + 0.4 * typ + rng.normal(0, 0.05, 400)
    s = typicality_decay_stats(typ, np.clip(agr, 0, 1))
    assert s["typicality_agreement_corr"] > 0.5
    assert s["typical_minus_atypical_agreement"] > 0.2


def test_classify_scaling():
    assert classify_scaling([0.0, 0.01, 0.02], [0.5, 0.4, 0.3]) == "scaling_tacit"
    assert classify_scaling([0.02, 0.15, 0.3], [0.5, 0.4, 0.3]) == "eventually_articulable"
    assert classify_scaling([0.07, 0.08], [0.5, 0.4]) == "intermediate"
    assert classify_scaling([0.0, 0.0], [0.05, 0.08]) == "no_gap_anywhere"


def _grid(tmp_path, job, domain, rows):
    d = tmp_path / job
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"grid_{domain}_fix_rep0.npz",
        scores=np.vstack([v for _m, v in rows]),
        meta=np.array([json.dumps(m) for m, _v in rows], dtype=object))


def test_w0_battery_on_synthetic_slice(tmp_path):
    """End-to-end: synthetic target+executor grids -> run all W0 probes -> rows sane."""
    rng = np.random.default_rng(7)
    n_items, cells = 60, [f"TB::humor::c{i}" for i in range(12)]
    m = lambda c, a, f="canonical", cf=None: {
        "cell_id": c, "arm_id": a, "form": f, "control_for": cf}
    tgt_rows, exe_rows, exe2_rows = [], [], []
    for i, c in enumerate(cells):
        t = rng.normal(size=n_items)
        for f in ("canonical", "question"):
            tgt_rows.append((m(c, "name", f), t + rng.normal(scale=0.1, size=n_items)))
        # executor: half the cells articulation-rescuable, half not
        weak = t + rng.normal(scale=2.0, size=n_items)
        artic = (t + rng.normal(scale=0.3, size=n_items)) if i < 6 \
            else rng.normal(size=n_items)
        exe_rows.append((m(c, "name"), weak))
        exe_rows.append((m(c, "source_definition"), artic))
        exe_rows.append((m(c, "control_inert_definition", cf="source_definition"),
                         rng.normal(size=n_items)))
        # 6 filler articulation variants so the subspace cap has >=8 prompt rows/cell
        for k in range(6):
            exe_rows.append((m(c, f"filler_arm_{k}"),
                             t * rng.uniform(0.1, 0.6) + rng.normal(scale=1.0, size=n_items)))
        # second rung (stronger executor) so the scaling ladder has >=2 points
        exe2_rows.append((m(c, "name"), t + rng.normal(scale=1.0, size=n_items)))
        exe2_rows.append((m(c, "source_definition"),
                          (t + rng.normal(scale=0.2, size=n_items)) if i < 6
                          else rng.normal(size=n_items)))
    _grid(tmp_path, "fake_target", "humor", tgt_rows)
    _grid(tmp_path, "fake_8b_executor", "humor", exe_rows)
    _grid(tmp_path, "fake_14b_executor", "humor", exe2_rows)

    ctx = ArtifactContext(
        family="qwen25", domain="humor",
        scores_root_override=str(tmp_path), target_job_override="fake_target",
        items_override={"texts": [f"item {i}" for i in range(n_items)],
                        "hashes": [f"h{i}" for i in range(n_items)]},
        typicality_override=rng.uniform(0, 1, n_items))

    from methods.tacit_channels.battery.profile import convergence, run_battery
    rows, executed = run_battery(ctx, wave=0, run_tag="test")
    assert len(rows) > 100
    probes_seen = {r["probe"] for r in rows}
    assert {"P-CHAN-core", "P-STAT-1", "P-GT-1", "P-GT-3", "P-GEN-1",
            "P-SCAL-1", "P-SCAL-2", "P-SCAL-3", "P-CEIL-1"} <= probes_seen
    for _pid, res in executed.items():
        assert not (isinstance(res, str) and res.startswith("ERROR")), (_pid, res)
    conv = convergence(rows)
    assert conv["n_probes"] >= 4


def test_item_agreement_bounds():
    t = np.array([0.1, 0.5, 0.9, 0.3])
    assert np.allclose(item_agreement(t, t), 1.0)
    rev = item_agreement(-t, t)
    assert rev.mean() < 0.5


def test_pass_planner_variants():
    from methods.tacit_channels.battery.passes import (
        build_single_stage_rows, plan_summary, assemble_reason_first_tf,
    )
    cells = {
        "c1": {"arms": [{"id": "name", "forms": [
            {"id": "canonical", "prompt": "Wit"},
            {"id": "question", "prompt": "Does it satisfy? Wit"}]}]},
        "c2": {"arms": [{"id": "name", "forms": [
            {"id": "canonical", "prompt": "Timing"}]}]},
    }
    rows = build_single_stage_rows(
        cells, ("tf", "exclusion", "negated", "composed"),
        composed_pairs=(("c1", "c2"),), forms=("canonical", "question"))
    s = plan_summary(rows)
    assert s["tf"] == 3 and s["exclusion"] == 3 and s["negated"] == 3
    assert s["composed"] == 1  # only canonical form exists for both members
    exc = next(r for r in rows if r["variant"] == "exclusion" and r["cell_id"] == "c1")
    assert "AGAINST" in exc["content"] and "Wit" in exc["content"]
    comp = next(r for r in rows if r["variant"] == "composed")
    assert "SIMULTANEOUSLY" in comp["content"] and comp["pair"] == ["c1", "c2"]
    tpl = "Criterion:\n{rubric}\n\nText:\n{text}\n\nAnswer with exactly one word: YES or NO."
    tf2 = assemble_reason_first_tf(tpl, rows[0], "a joke", "it lands cleanly", 100)
    assert "Reasoning: it lands cleanly" in tf2 and tf2.endswith("YES or NO.")


def test_variant_pass_runner_scoring_and_acceptance(tmp_path):
    """score_rows with a stub scorer: arm_id encoding, meta fields, acceptance matching."""
    from methods.tacit_channels.battery.passes import build_holistic_row, build_single_stage_rows
    from methods.tacit_channels.battery.run_variant_pass import run_acceptance, score_rows

    cells = {
        "c1": {"arms": [{"id": "name", "forms": [{"id": "canonical", "prompt": "Wit"}]}]},
        "c2": {"arms": [{"id": "name", "forms": [{"id": "canonical", "prompt": "Timing"}]}]},
    }
    rows = build_single_stage_rows(cells, ("tf", "exclusion", "negated", "composed"),
                                   composed_pairs=(("c1", "c2"),), forms=("canonical",))
    rows.append(build_holistic_row("humor"))
    texts = [f"item {i}" for i in range(8)]
    tpl = "Criterion:\n{rubric}\n\nText:\n{text}\n\nAnswer with exactly one word: YES or NO."

    def stub(_backend, prompts, pos, neg, expected_token_ids, seed):
        assert expected_token_ids == {"YES": 1, "NO": 2}
        # deterministic per-row vector keyed by prompt content
        base = (hash(prompts[0]) % 97) / 97.0
        return [(base + i * 0.01) % 1.0 for i in range(len(prompts))]

    matrix, meta = score_rows(None, rows, texts, tpl, 100, {"YES": 1, "NO": 2}, "humor",
                              score_fn=stub)
    assert matrix.shape == (len(rows), 8)
    arm_ids = {m["arm_id"] for m in meta}
    assert arm_ids == {"name", "name_exclusion", "name_negated", "name_composed", "holistic"}
    comp = next(m for m in meta if m["arm_id"] == "name_composed")
    assert comp["cell_id"] == "c1&&c2" and comp["pair"] == ["c1", "c2"]
    hol = next(m for m in meta if m["arm_id"] == "holistic")
    assert hol["cell_id"] == "HOLISTIC::humor"

    # acceptance: reference grid whose name rows equal the runner's tf rows -> PASS
    tf_idx = [i for i, m in enumerate(meta) if m["variant"] == "tf"]
    ref_meta = [json.dumps({"cell_id": meta[i]["cell_id"], "arm_id": "name",
                            "form": meta[i]["form"]}) for i in tf_idx]
    ref = tmp_path / "ref.npz"
    np.savez_compressed(ref, scores=matrix[tf_idx],
                        meta=np.array(ref_meta, dtype=object))
    run_acceptance(matrix, meta, str(ref), 0.999)

    # perturbed reference -> FAIL
    rng = np.random.default_rng(0)
    np.savez_compressed(tmp_path / "bad.npz",
                        scores=rng.normal(size=matrix[tf_idx].shape),
                        meta=np.array(ref_meta, dtype=object))
    with pytest.raises(SystemExit):
        run_acceptance(matrix, meta, str(tmp_path / "bad.npz"), 0.999)


def test_w1b_two_stage_helpers(tmp_path):
    from methods.tacit_channels.battery.run_reason_first_pass import (
        parse_confidence, reason_first_generation_prompt, reason_first_tf_prompt,
        run_confidence, run_reason_first, tf_answers_from_grid,
    )
    tpl = "Criterion:\n{rubric}\n\nText:\n{text}\n\nAnswer with exactly one word: YES or NO."
    row = {"cell_id": "c1", "variant": "tf", "form": "canonical", "content": "Wit"}
    g = reason_first_generation_prompt(tpl, row, "a joke", 100)
    assert "criterion-relevant considerations" in g and "exactly one word" not in g
    tf = reason_first_tf_prompt(tpl, row, "a joke", "  it\nlands   cleanly ", 100)
    assert "Reasoning: it lands cleanly" in tf and tf.endswith("YES or NO.")

    assert parse_confidence("I'd say 85 out of 100") == 85.0
    assert parse_confidence("confidence: 100.") == 100.0
    assert np.isnan(parse_confidence("pretty sure"))
    assert np.isnan(parse_confidence("about 250"))
    assert parse_confidence("7") == 7.0

    # tf-answer derivation from a W1a-style grid
    meta_rows = [json.dumps({"cell_id": "c1", "variant": "tf", "form": "canonical",
                             "domain": "humor", "arm_id": "name"}),
                 json.dumps({"cell_id": "c1", "variant": "negated", "form": "canonical",
                             "domain": "humor", "arm_id": "name_negated"})]
    np.savez_compressed(tmp_path / "tf.npz",
                        scores=np.array([[0.9, 0.2, 0.5], [0.1, 0.1, 0.1]]),
                        meta=np.array(meta_rows, dtype=object))
    answers = tf_answers_from_grid(str(tmp_path / "tf.npz"), "humor")
    assert list(answers) == [("c1", "canonical")]
    assert answers[("c1", "canonical")].tolist() == [True, False, True]

    # end-to-end with stub gen/score fns
    texts = ["t0", "t1", "t2"]
    rows = [dict(row, domain="humor")]
    gen = lambda prompts, seed: [f"rationale {i}" for i in range(len(prompts))]
    score = lambda _b, prompts, pos, neg, expected_token_ids, seed: [0.6] * len(prompts)
    m, meta2, rats = run_reason_first(None, rows, texts, tpl, 100, {"YES": 1, "NO": 2},
                                      gen_max_tokens=16, score_fn=score, gen_fn=gen)
    assert m.shape == (1, 3) and meta2[0]["arm_id"] == "name_reason_first"
    assert rats[0]["rationales"] == ["rationale 0", "rationale 1", "rationale 2"]
    cgen = lambda prompts, seed: ["90" if "Your answer was: YES" in p else "10"
                                  for p in prompts]
    conf, cmeta, rates = run_confidence(None, rows, texts, tpl, 100, answers, gen_fn=cgen)
    assert conf.shape == (1, 3) and conf[0].tolist() == [90.0, 10.0, 90.0]
    assert rates == [1.0] and cmeta[0]["arm_id"] == "name_confidence"

"""Tests for deep metric programs: safe exec, step-synchronous execution, flattening."""
import numpy as np
import pytest

from metrics_tree_infilling.deep_metrics import (
    DeepMetricProgram, execute_program, flatten_program, safe_exec)


def test_safe_exec_basic():
    assert safe_exec("out = s1 * 2", {"s1": 0.4}) == pytest.approx(0.8)
    assert safe_exec("out = min(1, len(s2) / 10)", {"s2": "abcde"}) == pytest.approx(0.5)
    assert safe_exec("out = math.sqrt(s1)", {"s1": 0.25}) == pytest.approx(0.5)
    assert safe_exec("out = 1 if re.search('lemma', text) else 0", {"text": "a lemma"}) == 1


def test_safe_exec_blocks_dangerous():
    for bad in ["import os\nout = 1", "out = open('/etc/passwd')",
                "out = __import__('os')", "out = ().__class__",
                "exec('x')", "out = getattr(s1, 'x')"]:
        assert safe_exec(bad, {"s1": 1.0}) is None


def test_safe_exec_failure_returns_none():
    assert safe_exec("out = 1 / 0", {}) is None
    assert safe_exec("out = undefined_name", {}) is None


def _mock_judge(prompts):
    # score prompts: return 0.9 when the item text mentions 'proof', else 0.2.
    # extract prompts: return the first word of the text.
    out = []
    for p in prompts:
        if '"score"' in p:
            out.append('{"score": 0.9}' if "proof" in p else '{"score": 0.2}')
        else:
            word = p.split("TEXT:")[-1].strip().split()[0] if "TEXT:" in p else "x"
            out.append('{"span": "%s"}' % word)
    return out


def test_execute_program_mixed_steps():
    prog = DeepMetricProgram.from_json({
        "name": "verify-claim", "description": "extract then check",
        "steps": [
            {"id": "s1", "kind": "judge_extract", "prompt": "TEXT: {text}\nExtract first word."},
            {"id": "s2", "kind": "judge_score", "prompt": "TEXT: {text}\nClaim: {s1}. Score."},
            {"id": "s3", "kind": "code", "code": "out = s2 * (1 if len(s1) > 0 else 0)"},
        ],
        "aggregate": "out = s3"})
    texts = ["proof of the lemma", "just a comment"]
    scores = execute_program(prog, texts, _mock_judge)
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.2)
    assert prog.n_judge_steps == 2


def test_program_validation():
    with pytest.raises(ValueError):
        DeepMetricProgram.from_json({"name": "x", "steps": [], "aggregate": "out = 1"})
    with pytest.raises(ValueError):
        DeepMetricProgram.from_json({"name": "x", "steps": [
            {"id": "s1", "kind": "shell", "code": "out = 1"}], "aggregate": "out = s1"})


def test_bad_judge_output_gives_nan():
    prog = DeepMetricProgram.from_json({
        "name": "n", "description": "",
        "steps": [{"id": "s1", "kind": "judge_score", "prompt": "T: {text}"}],
        "aggregate": "out = s1"})
    scores = execute_program(prog, ["a"], lambda ps: ["garbage not json"])
    assert np.isnan(scores[0])


def test_flatten_program():
    prog = DeepMetricProgram.from_json({
        "name": "n", "description": "checks the key computation.",
        "steps": [{"id": "s1", "kind": "judge_extract", "prompt": "Extract from {text}"},
                  {"id": "s2", "kind": "code", "code": "out = len(s1)"}],
        "aggregate": "out = min(1, s2)"})
    flat = flatten_program(prog)
    assert "checks the key computation" in flat and "{text}" not in flat

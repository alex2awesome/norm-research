"""Apply a MetricArtifact to texts: PromptJudge (LLM) and CodeJudge (subprocess sandbox).

Both return ``(scores, applicable)`` aligned to the input texts; ``scores`` in [0,1] with
NaN where not applicable. The CodeJudge runs the artifact's module in ONE subprocess per
batch (texts streamed over stdin as JSON) with a hard timeout — generated code never
executes in the parent interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .artifact import MetricArtifact
from .backends import LLMBackend, parse_json_obj

_JUDGE_SYSTEM_TMPL = """You are a careful evaluator of {domain}. \
Apply the rubric below to the single item given. Respond ONLY with a JSON object: \
{{"score": <float in [0,1]>, "applicable": <true|false>}}. Use "applicable": false only \
when the rubric genuinely cannot be applied to this item."""

_JUDGE_TEMPLATE = """RUBRIC:
{rubric}

{item_label}:
```
{text}
```

JSON:"""


def judge_prompts(cfg) -> tuple:
    """(system prompt, item template) for the configured task framing — the only place
    item-type wording enters judge prompts (cross-task bug guard)."""
    system = _JUDGE_SYSTEM_TMPL.format(
        domain=getattr(cfg, "domain_plural", "competitive-programming solutions"))
    template = _JUDGE_TEMPLATE.replace(
        "{item_label}", getattr(cfg, "item_label", "SOLUTION"))
    return system, template


def _valid_judge_output(raw: str) -> bool:
    obj = parse_json_obj(raw)
    if not obj or "score" not in obj:
        return False
    try:
        s = float(obj["score"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= s <= 1.0


class PromptJudge:
    def __init__(self, backend: LLMBackend, cfg):
        self.backend = backend
        self.max_text_chars = getattr(cfg, "max_text_chars", 4000)
        self.system, self.template = judge_prompts(cfg)

    def score(self, artifact: MetricArtifact, texts: List[str],
              temperature: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        prompts = [self.template.format(rubric=artifact.body,
                                        text=t[: self.max_text_chars]) for t in texts]
        raws = self.backend.generate_batch(prompts, system=self.system, max_tokens=80,
                                           validate=_valid_judge_output,
                                           temperature=temperature)
        scores = np.full(len(texts), np.nan)
        applicable = np.zeros(len(texts), dtype=bool)
        for i, raw in enumerate(raws):
            obj = parse_json_obj(raw)
            if not obj:
                continue
            try:
                s = float(obj.get("score"))
            except (TypeError, ValueError):
                continue
            if not (0.0 <= s <= 1.0):
                continue
            if obj.get("applicable") is False:
                continue
            scores[i], applicable[i] = s, True
        return scores, applicable


_RUNNER = r"""
import json, sys, math
src = sys.argv[1]
ns = {}
exec(compile(open(src).read(), "metric_code", "exec"), ns)
score = ns.get("score")
out = []
for t in json.load(sys.stdin):
    try:
        v = score(t)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        else:
            out.append(max(0.0, min(1.0, float(v))))
    except Exception:
        out.append(None)
print(json.dumps(out))
"""


class CodeJudge:
    def __init__(self, timeout_s: int = 30):
        self.timeout_s = timeout_s

    def score(self, artifact: MetricArtifact, texts: List[str],
              temperature=None) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.full(len(texts), np.nan)
        applicable = np.zeros(len(texts), dtype=bool)
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "metric_code.py"
            src.write_text(artifact.body)
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _RUNNER, str(src)],
                    input=json.dumps(texts), capture_output=True, text=True,
                    timeout=self.timeout_s)
                vals = json.loads(proc.stdout.strip() or "[]")
            except Exception:
                return scores, applicable
        for i, v in enumerate(vals[: len(texts)]):
            if v is not None:
                scores[i], applicable[i] = float(v), True
        return scores, applicable


def make_judge(artifact: MetricArtifact, roles, cfg):
    if artifact.kind == "code":
        return CodeJudge()
    return PromptJudge(roles.judge, cfg)

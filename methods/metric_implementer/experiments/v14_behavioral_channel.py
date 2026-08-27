"""v14 behavioral induce-and-execute channel with cell-granular reuse."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import sys
from typing import Mapping, Sequence

import numpy as np

from ..backends import parse_json_obj
from ..batch_scoring import _YESNO_TEMPLATE
from ..recon_channel import _feat_corr_table
from .cr3_evidence_store import EvidenceCellStore
from .v14_value_bound import (
    balanced_agreement,
    binary_entropy_bits,
    enumerate_states,
    plugin_binary_mutual_information,
)


BEHAVIORAL_ARMS = ("unconstrained", "no_verbatim_examples")
STATE_TABLE_SCHEMA = "cr3-v14-behavioral-state-tables-v1"
EXECUTION_TEMPLATE_SHA256 = hashlib.sha256(_YESNO_TEMPLATE.encode("utf-8")).hexdigest()
DEFAULT_TEMPLATE = """You are reverse-engineering a hidden binary evaluation criterion applied to {noun}.
Infer one general rule from the labeled examples. Use the feature correlations as fallible clues,
compare hypotheses against every label, and state the most specific rule that generalizes.

FEATURE CORRELATIONS:
{feature_table}

LABELED EXAMPLES:
{examples}

{arm_instruction}Reply with only the criterion."""
BLIND_TEMPLATE = """Infer a plausible binary criterion for {noun} without seeing labeled examples.
{arm_instruction}Reply with only the criterion."""


def canonical_template_sha256(template: str) -> str:
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def normalized_rule(raw: str) -> str:
    text = str(raw or "").strip()
    obj = parse_json_obj(text)
    if obj:
        for field in ("rule", "rubric", "criterion"):
            if str(obj.get(field) or "").strip():
                text = str(obj[field]).strip()
                break
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        raise RuntimeError("decoder returned an empty induced rule")
    return re.sub(r"\s+", " ", text)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", str(text))


def _lower_words(text: str) -> list[str]:
    return [word.lower() for word in _words(text)]


def _shared_shingle(rule: str, demos: Sequence[str], n: int = 8) -> tuple[str, ...] | None:
    words = _lower_words(rule)
    if len(words) < n:
        return None
    shingles = {tuple(words[index:index + n]) for index in range(len(words) - n + 1)}
    for demo in demos:
        demo_words = _lower_words(demo)
        for index in range(len(demo_words) - n + 1):
            shingle = tuple(demo_words[index:index + n])
            if shingle in shingles:
                return shingle
    return None


def _lifted_proper_nouns(rule: str, demos: Sequence[str]) -> set[str]:
    rule_tokens = set(re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", rule))
    lifted = set()
    for demo in demos:
        # Ignore a capitalized token at sentence start; it is not evidence of a name.
        tokens = set(re.findall(r"(?<![.!?]\s)\b[A-Z][A-Za-z0-9_-]+\b", str(demo)))
        lifted.update(rule_tokens.intersection(tokens))
    return lifted


def _quoted_spans(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", value.strip().lower())
        for value in re.findall(r"[\"“](.*?)[\"”]|'(.*?)'", str(text))
        for value in (value if isinstance(value, tuple) else (value,))
        if str(value).strip()
    }


def no_verbatim_violations(
    rule: str, demos: Sequence[str], *, corpus_token_counts: Mapping[str, int],
    max_tokens: int = 64, rare_threshold: int = 5, with_details: bool = False,
) -> list[str] | tuple[list[str], list[str]]:
    """Return fail-closed structural violations for the rule-only arm.

    With ``with_details=True`` also return human-readable evidence strings naming
    the offending content, so a bounded repair prompt can target it instead of
    rewriting blind.
    """
    violations = []
    details = []
    words = _words(rule)
    if len(words) > int(max_tokens):
        violations.append("token_cap")
        details.append(f"token_cap: {len(words)} words exceeds the {int(max_tokens)}-word maximum")
    if not rule.rstrip().endswith("?") or rule.count("?") != 1:
        violations.append("single_question_format")
        details.append("single_question_format: reply with exactly one question ending in a single '?'")
    shingle = _shared_shingle(rule, demos, 8)
    if shingle is not None:
        violations.append("eight_word_demo_shingle")
        details.append("eight_word_demo_shingle: remove the copied phrase \"" + " ".join(shingle) + "\"")
    lifted_nouns = _lifted_proper_nouns(rule, demos)
    if lifted_nouns:
        violations.append("lifted_proper_noun")
        details.append("lifted_proper_noun: do not use " + ", ".join(sorted(lifted_nouns)[:8]))
    rule_quotes = _quoted_spans(rule)
    demo_quotes = set().union(*(_quoted_spans(text) for text in demos)) if demos else set()
    lifted_quotes = rule_quotes.intersection(demo_quotes)
    if lifted_quotes:
        violations.append("lifted_quote")
        details.append(
            "lifted_quote: remove the quoted span(s) "
            + "; ".join(f'"{value}"' for value in sorted(lifted_quotes)[:4])
        )
    demo_tokens = set(token for text in demos for token in _lower_words(text))
    lifted_rare = {
        token for token in _lower_words(rule)
        if token in demo_tokens and int(corpus_token_counts.get(token, 0)) < int(rare_threshold)
        and len(token) >= 4
    }
    if lifted_rare:
        violations.append("lifted_rare_token")
        details.append(
            "lifted_rare_token: replace the corpus-specific word(s) "
            + ", ".join(sorted(lifted_rare)[:12])
            + " with general vocabulary"
        )
    if with_details:
        return sorted(set(violations)), details
    return sorted(set(violations))


def corpus_token_counts(texts: Sequence[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(_lower_words(text))
    return counts


def format_examples(texts: Sequence[str], labels: Sequence[int], max_chars: int) -> str:
    return "\n\n".join(
        f"[label={int(label)}]\n```\n{str(text)[:int(max_chars)]}\n```"
        for text, label in zip(texts, labels)
    )


def induction_prompt(
    *, template: str, noun: str, texts: Sequence[str], labels: Sequence[int],
    max_chars: int, arm: str,
) -> str:
    feature_table, _ = _feat_corr_table(list(texts), np.asarray(labels, dtype=float))
    arm_instruction = (
        "Write one short rubric question. Do not copy, quote, name, or carry any example; "
        "describe only a general rule for unseen texts. "
        if arm == "no_verbatim_examples" else ""
    )
    return str(template).format(
        noun=str(noun), feature_table=feature_table,
        examples=format_examples(texts, labels, int(max_chars)),
        arm_instruction=arm_instruction,
    )


def blind_prompt(*, template: str, noun: str, arm: str) -> str:
    instruction = (
        "Write one short rubric question without examples, names, quotations, or corpus-specific tokens. "
        if arm == "no_verbatim_examples" else ""
    )
    return str(template).format(
        noun=str(noun),
        feature_table="(No labeled examples or feature correlations are provided.)",
        examples="(No labeled examples are provided.)",
        arm_instruction=instruction,
    )


def _repair_prompt(rule: str, violations: Sequence[str], details: Sequence[str] = ()) -> str:
    guidance = "\n".join(f"- {line}" for line in details) if details else (
        "- " + ", ".join(violations)
    )
    return (
        "Rewrite the candidate below as exactly one short general rubric question, maximum 64 "
        "tokens. Do not quote or copy examples, proper nouns, or rare corpus-specific words. "
        "Keep the same evaluative meaning while fixing every issue listed:\n"
        f"{guidance}\n"
        "Reply only with the question.\n\n"
        f"CANDIDATE:\n{rule}"
    )


def _seed(key: str, attempt: int = 0) -> int:
    digest = hashlib.sha256(f"{key}\x1f{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _generate(constructor, prompts: Sequence[str], seeds: Sequence[int], *, max_tokens: int = 128) -> list[str]:
    if not prompts:
        return []
    output = constructor.generate_batch(
        list(prompts), system=None, max_tokens=int(max_tokens), temperature=0.0,
        seed=list(map(int, seeds)),
    )
    if len(output) != len(prompts):
        raise RuntimeError("decoder returned an incomplete induction batch")
    return [normalized_rule(value) for value in output]


def induce_requests(
    constructor, *, requests: Sequence[Mapping[str, object]], store: EvidenceCellStore,
    corpus_counts: Mapping[str, int], max_repairs: int = 4,
) -> dict[str, dict]:
    """Induce only missing cells, with bounded batched no-verbatim repair.

    A cell whose rule still violates the no-verbatim constraints after
    ``max_repairs`` targeted repair rounds is stored as a VOID artifact
    (fail closed, recorded) rather than aborting the stage: one criterion's
    unparaphrasable vocabulary must not destroy the campaign. Downstream
    consumers must skip rows carrying ``"void": True``.
    """
    output = {}
    missing = []
    for request in requests:
        key = str(request["cache_key"])
        row = store.get(key)
        if row is None:
            missing.append(request)
        else:
            output[key] = row
    rules = _generate(
        constructor, [str(row["prompt"]) for row in missing],
        [_seed(str(row["cache_key"])) for row in missing],
    )
    by_key = {str(request["cache_key"]): request for request in missing}
    current = {
        str(request["cache_key"]): rule for request, rule in zip(missing, rules)
    }
    attempts = {key: 0 for key in by_key}

    def _check(key: str) -> tuple[list[str], list[str]]:
        request = by_key[key]
        if request["arm"] != "no_verbatim_examples":
            return [], []
        return no_verbatim_violations(
            current[key], request["example_texts"],
            corpus_token_counts=corpus_counts, with_details=True,
        )

    pending = {key: found for key in by_key if (found := _check(key))[0]}
    for _ in range(int(max_repairs)):
        if not pending:
            break
        round_keys = sorted(pending)
        repaired = _generate(
            constructor,
            [_repair_prompt(current[key], *pending[key]) for key in round_keys],
            [_seed(key, attempts[key] + 1) for key in round_keys],
        )
        next_pending = {}
        for key, rule in zip(round_keys, repaired):
            attempts[key] += 1
            current[key] = rule
            found = _check(key)
            if found[0]:
                next_pending[key] = found
        pending = next_pending

    for key, request in by_key.items():
        base = {
            "arm": str(request["arm"]),
            "panel_sha256": str(request["panel_sha256"]),
            "state": int(request["state"]),
            "template_sha256": str(request["template_sha256"]),
            "no_verbatim_enforced": request["arm"] == "no_verbatim_examples",
            "repair_attempts": int(attempts[key]),
        }
        if key in pending:
            payload = store.put(key, "induction", {
                **base, "rule": "", "rule_sha256": "",
                "void": True, "void_violations": list(pending[key][0]),
            })
        else:
            rule = current[key]
            payload = store.put(key, "induction", {
                **base, "rule": rule,
                "rule_sha256": hashlib.sha256(rule.encode("utf-8")).hexdigest(),
            })
        output[key] = payload
    return output


def execute_rule_probe_cells(
    executor, *, rules: Mapping[str, str], probe_texts: Sequence[str],
    executor_revision: str, readout_id: str, store: EvidenceCellStore,
    max_chars: int, query_batch_size: int = 2048,
) -> dict[tuple[str, int], dict]:
    """Execute missing rule/probe cells and return a dense logical view."""
    rows = {}
    missing = []
    for rule_sha, rule in rules.items():
        if hashlib.sha256(str(rule).encode("utf-8")).hexdigest() != str(rule_sha):
            raise ValueError("rule text does not match its content SHA")
        for index, text in enumerate(probe_texts):
            probe_sha = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            key = store.rule_probe_key(
                rule_sha256=rule_sha, probe_sha256=probe_sha,
                executor_revision=str(executor_revision), readout_id=str(readout_id),
                execution_template_sha256=EXECUTION_TEMPLATE_SHA256,
            )
            cached = store.get(key)
            if cached is None:
                missing.append((rule_sha, rule, index, str(text), probe_sha, key))
            else:
                rows[(rule_sha, index)] = cached
    for start in range(0, len(missing), int(query_batch_size)):
        batch = missing[start:start + int(query_batch_size)]
        prompts = [
            _YESNO_TEMPLATE.format(rubric=rule, text=text[:int(max_chars)])
            for _, rule, _, text, _, _ in batch
        ]
        scores = np.asarray(executor.score_binary_constrained(
            prompts, system=None, pos="YES", neg="NO", seed=0,
        ), dtype=float)
        if scores.shape != (len(batch),) or np.any(~np.isfinite(scores)):
            raise RuntimeError("executor returned incomplete or non-finite rule/probe cells")
        for item, score in zip(batch, scores):
            rule_sha, _, index, _, probe_sha, key = item
            # Concurrent shards can execute the same (rule, probe) cell (shared
            # control states) independently; vLLM batching is not bit-exact
            # across different batch compositions, so raw floats can differ in
            # the last few digits between shards. Quantize before hashing so
            # equivalent cells agree instead of tripping the disagreement guard.
            p_yes = round(float(score), 6)
            try:
                payload = store.put(key, "rule_probe", {
                    "rule_sha256": rule_sha,
                    "probe_sha256": probe_sha,
                    "p_yes": p_yes,
                    "hard_prediction": int(p_yes > 0.5),
                    "executor_revision": str(executor_revision),
                    "readout_id": str(readout_id),
                })
            except RuntimeError:
                # Two shards raced the same missing cell. Had this shard polled
                # the store a moment later it would have taken the cached value
                # via the get() above — so accepting the first-written cell is
                # identical to the ordinary cache-hit path and keeps the frozen
                # instrument value stable. vLLM scores drift with batch
                # composition (deltas up to a few percent observed), so exact
                # re-execution agreement is not a valid invariant here. Every
                # collision is logged with its delta for post-hoc audit.
                stored = store.get(key)
                if stored is None:
                    raise
                delta = abs(float(stored["p_yes"]) - p_yes)
                print(
                    f"[rule_probe collision] key={key[:16]} stored={stored['p_yes']:.6f} "
                    f"ours={p_yes:.6f} delta={delta:.6f} "
                    f"hard_flip={int(stored['hard_prediction']) != int(p_yes > 0.5)}",
                    file=sys.stderr, flush=True,
                )
                payload = stored
            rows[(rule_sha, index)] = payload
    return rows


def shuffled_state(state: int, panel_size: int, panel_sha256: str) -> int:
    bits = enumerate_states(panel_size)[int(state)]
    if np.unique(bits).size < 2:
        return int(state)
    shift = 1 + int(str(panel_sha256)[:8], 16) % (panel_size - 1)
    for offset in [shift, *range(1, panel_size)]:
        candidate = np.roll(bits, offset)
        if not np.array_equal(candidate, bits):
            weights = 1 << np.arange(panel_size - 1, -1, -1)
            return int(np.sum(candidate.astype(int) * weights))
    return int(state)


def evaluate_behavioral_state_tables_v14(
    constructor, executor=None, *, design_manifest: Mapping[str, object],
    probe_texts: Sequence[str], heldout_indices: Sequence[int],
    heldout_target: Sequence[int], noun: str, decoder_revision: str,
    executor_revision: str, readout_id: str, store: EvidenceCellStore,
    templates: Mapping[str, str] | None = None, max_chars: int = 600,
    query_batch_size: int = 2048, induction_only: bool = False,
    probe_embeddings: np.ndarray | None = None,
    state_indices_by_panel: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Value exhaustive CERT states or the declared observed-only FAST states."""
    panels = list(design_manifest["panels"])
    panel_size = int(design_manifest["panel_size"])
    if panel_size not in {6, 8}:
        raise ValueError("v14 behavioral supports frozen k=6 fan-out and k=8 sentinels")
    n_states = 1 << panel_size
    heldout = list(map(int, heldout_indices))
    teaching = set(map(int, design_manifest["teaching_indices"]))
    if teaching.intersection(heldout):
        raise RuntimeError("teaching and held-out probes overlap")
    target = np.asarray(heldout_target, dtype=np.uint8)
    if target.shape != (len(heldout),):
        raise ValueError("heldout target is not aligned")
    h_texts = [str(probe_texts[index]) for index in heldout]
    transfer_strata = None
    if probe_embeddings is not None:
        embeddings = np.asarray(probe_embeddings, dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(probe_texts):
            raise ValueError("probe embeddings do not align with the probe bank")
        stratum_size = min(50, len(heldout) // 2)
        if stratum_size < 1:
            raise ValueError("heldout set is too small for near/far transfer strata")
        transfer_strata = [
            near_far_indices(
                embeddings[np.asarray(panel["indices"], dtype=int)],
                embeddings[np.asarray(heldout, dtype=int)],
                stratum_size=stratum_size,
            )
            for panel in panels
        ]
    counts = corpus_token_counts(probe_texts)
    template_map = {arm: DEFAULT_TEMPLATE for arm in BEHAVIORAL_ARMS}
    if templates:
        template_map.update({str(key): str(value) for key, value in templates.items()})

    requests = []
    request_keys = {}
    states = enumerate_states(panel_size)
    requested_by_panel = []
    scored_by_panel = []
    for panel_position, panel in enumerate(panels):
        requested = (
            set(range(n_states)) if state_indices_by_panel is None
            else set(map(int, state_indices_by_panel[panel_position]))
        )
        if not requested or min(requested) < 0 or max(requested) >= n_states:
            raise ValueError("behavioral requested state lies outside the panel code space")
        # Only the originally requested lane states are SCORED. The shuffled control
        # partners are appended to the induction set so each scored state's control
        # rule exists, but they are not themselves scored: shuffled_state is a bit
        # rotation (not an involution), so the control of a control generally lies
        # outside the requested set and must not be demanded by the fill loop.
        scored_by_panel.append(sorted(requested))
        requested.update(
            shuffled_state(state, panel_size, str(panel["panel_sha256"]))
            for state in list(requested)
        )
        requested_by_panel.append(sorted(requested))
    for arm in BEHAVIORAL_ARMS:
        template_sha = canonical_template_sha256(template_map[arm])
        for panel_position, panel in enumerate(panels):
            indices = list(map(int, panel["indices"]))
            texts = [str(probe_texts[index]) for index in indices]
            for state in requested_by_panel[panel_position]:
                labels = states[state]
                prompt = induction_prompt(
                    template=template_map[arm], noun=noun, texts=texts,
                    labels=labels.tolist(), max_chars=max_chars, arm=arm,
                )
                key = store.induction_key(
                    template_sha256=template_sha, decoder_revision=decoder_revision,
                    arm=arm, panel_sha256=str(panel["panel_sha256"]), state=state,
                )
                requests.append({
                    "cache_key": key, "prompt": prompt, "arm": arm,
                    "panel_sha256": str(panel["panel_sha256"]), "state": state,
                    "template_sha256": template_sha, "example_texts": texts,
                })
                request_keys[(arm, panel_position, state)] = key
        blind_sha = canonical_template_sha256("blind\x1f" + template_map[arm])
        blind_panel_sha = "blind:" + hashlib.sha256(
            str(noun).encode("utf-8")
        ).hexdigest()
        blind_key = store.induction_key(
            template_sha256=blind_sha, decoder_revision=decoder_revision,
            arm=arm, panel_sha256=blind_panel_sha, state=-1,
        )
        requests.append({
            "cache_key": blind_key,
            "prompt": blind_prompt(template=template_map[arm], noun=noun, arm=arm),
            "arm": arm, "panel_sha256": blind_panel_sha, "state": -1,
            "template_sha256": blind_sha, "example_texts": [],
        })
        request_keys[(arm, -1, -1)] = blind_key
    induced = induce_requests(
        constructor, requests=requests, store=store, corpus_counts=counts,
    )
    rules = {
        str(row["rule_sha256"]): str(row["rule"])
        for row in induced.values()
        if not row.get("void") and str(row.get("rule_sha256", ""))
    }
    n_void = sum(1 for row in induced.values() if row.get("void"))
    if induction_only:
        return {
            "schema": STATE_TABLE_SCHEMA,
            "channel": "behavioral",
            "stage": "induction_complete",
            "panel_design_sha256": str(design_manifest["design_sha256"]),
            "decoder_revision": str(decoder_revision),
            "n_induction_cells": len(induced),
            "n_void_induction_cells": int(n_void),
            "n_distinct_rules": len(rules),
        }
    if executor is None:
        raise ValueError("executor is required after the induction-only phase")
    executions = execute_rule_probe_cells(
        executor, rules=rules, probe_texts=h_texts, executor_revision=executor_revision,
        readout_id=readout_id, store=store, max_chars=max_chars,
        query_batch_size=query_batch_size,
    )

    arms = {}
    for arm in BEHAVIORAL_ARMS:
        raw_mi = np.full((len(panels), n_states), np.nan, dtype=float)
        balanced = np.full_like(raw_mi, np.nan)
        shuffled_mi = np.full_like(raw_mi, np.nan)
        raw_lift = np.full_like(raw_mi, np.nan)
        rule_sha_table = np.full((len(panels), n_states), "", dtype="U64")
        hard_predictions = np.full(
            (len(panels), n_states, len(h_texts)), -1, dtype=np.int8,
        )
        near_lift = np.full_like(raw_mi, np.nan) if transfer_strata is not None else None
        far_lift = np.full_like(raw_mi, np.nan) if transfer_strata is not None else None
        blind_row = induced[request_keys[(arm, -1, -1)]]
        if blind_row.get("void"):
            # The blind control sees no demos, so only format constraints can void
            # it; a void blind row is an instrument failure, not a metric property.
            raise RuntimeError(
                f"blind induction voided for arm {arm}: "
                + ",".join(map(str, blind_row.get("void_violations", [])))
            )
        blind_sha = str(blind_row["rule_sha256"])
        blind_prediction = np.asarray([
            executions[(blind_sha, index)]["hard_prediction"] for index in range(len(h_texts))
        ], dtype=np.uint8)
        blind_mi = plugin_binary_mutual_information(target, blind_prediction)
        for panel_position, panel in enumerate(panels):
            if transfer_strata is not None:
                near_positions = transfer_strata[panel_position]["near"]
                far_positions = transfer_strata[panel_position]["far"]
                blind_near = plugin_binary_mutual_information(
                    target[near_positions], blind_prediction[near_positions]
                )
                blind_far = plugin_binary_mutual_information(
                    target[far_positions], blind_prediction[far_positions]
                )
            for state in scored_by_panel[panel_position]:
                row = induced[request_keys[(arm, panel_position, state)]]
                shuffled = shuffled_state(state, panel_size, str(panel["panel_sha256"]))
                shuffled_row = induced[request_keys[(arm, panel_position, shuffled)]]
                if row.get("void") or shuffled_row.get("void"):
                    # Void cells stay NaN and drop out of observed_state_mask.
                    continue
                rule_sha = str(row["rule_sha256"])
                shuffled_sha = str(shuffled_row["rule_sha256"])
                prediction = np.asarray([
                    executions[(rule_sha, index)]["hard_prediction"]
                    for index in range(len(h_texts))
                ], dtype=np.uint8)
                hard_predictions[panel_position, state] = prediction
                shuffled_prediction = np.asarray([
                    executions[(shuffled_sha, index)]["hard_prediction"]
                    for index in range(len(h_texts))
                ], dtype=np.uint8)
                # The aggregate's permutation null recomputes mi[control] from this
                # table, so the shuffled partner's predictions must be persisted
                # too (they are induced+executed above but are not lane states).
                hard_predictions[panel_position, shuffled] = shuffled_prediction
                raw_mi[panel_position, state] = plugin_binary_mutual_information(target, prediction)
                balanced[panel_position, state] = balanced_agreement(target, prediction)
                shuffled_mi[panel_position, state] = plugin_binary_mutual_information(
                    target, shuffled_prediction
                )
                raw_lift[panel_position, state] = raw_mi[panel_position, state] - max(
                    blind_mi, shuffled_mi[panel_position, state]
                )
                if transfer_strata is not None:
                    transmission_near = plugin_binary_mutual_information(
                        target[near_positions], prediction[near_positions]
                    )
                    transmission_far = plugin_binary_mutual_information(
                        target[far_positions], prediction[far_positions]
                    )
                    shuffled_near = plugin_binary_mutual_information(
                        target[near_positions], shuffled_prediction[near_positions]
                    )
                    shuffled_far = plugin_binary_mutual_information(
                        target[far_positions], shuffled_prediction[far_positions]
                    )
                    near_lift[panel_position, state] = transmission_near - max(
                        blind_near, shuffled_near
                    )
                    far_lift[panel_position, state] = transmission_far - max(
                        blind_far, shuffled_far
                    )
                rule_sha_table[panel_position, state] = rule_sha
        arm_result = {
            "raw_lift": raw_lift,
            "clipped_value": np.maximum(raw_lift, 0.0),
            "raw_mi": raw_mi,
            "shuffled_mi": shuffled_mi,
            "balanced_agreement": balanced,
            "blind_mi": float(blind_mi),
            "blind_balanced_agreement": balanced_agreement(target, blind_prediction),
            "target_entropy_bits": binary_entropy_bits(target),
            "rule_sha256": rule_sha_table,
            "hard_predictions": hard_predictions,
            "blind_hard_prediction": blind_prediction,
            "observed_state_mask": np.isfinite(raw_lift),
            "n_distinct_rules": int(len(set(
                value for value in rule_sha_table.ravel().tolist() if value
            ))),
            "n_void_induction_cells": int(sum(
                1 for (row_arm, _, _), cell_key in request_keys.items()
                if row_arm == arm and induced[cell_key].get("void")
            )),
            "template_sha256": canonical_template_sha256(template_map[arm]),
        }
        if transfer_strata is not None:
            arm_result.update({
                "near_raw_lift": near_lift,
                "far_raw_lift": far_lift,
                "near_clipped_value": np.maximum(near_lift, 0.0),
                "far_clipped_value": np.maximum(far_lift, 0.0),
            })
        arms[arm] = arm_result
    return {
        "schema": STATE_TABLE_SCHEMA,
        "channel": "behavioral",
        "panel_design_sha256": str(design_manifest["design_sha256"]),
        "decoder_revision": str(decoder_revision),
        "executor_revision": str(executor_revision),
        "readout_id": str(readout_id),
        "state_scope": "exhaustive" if state_indices_by_panel is None else "observed_only",
        "heldout_indices": heldout,
        "heldout_sha256": hashlib.sha256(json.dumps(heldout).encode("utf-8")).hexdigest(),
        "near_far_transfer": (
            None if transfer_strata is None else {
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "stratum_size": len(transfer_strata[0]["near"]),
                "panel_strata": [
                    {"near": row["near"], "far": row["far"]}
                    for row in transfer_strata
                ],
            }
        ),
        "arms": arms,
        "non_disclosure": {
            "candidate_prompt_text_passed_to_induction": False,
            "only_panel_texts_and_state_labels_enter_induction": True,
            "execution_uses_heldout_H_only": True,
        },
    }


def embedding_copy_penalty(rule_embedding: Sequence[float], demo_embeddings: np.ndarray) -> float:
    rule = np.asarray(rule_embedding, dtype=float)
    demos = np.asarray(demo_embeddings, dtype=float)
    if rule.ndim != 1 or demos.ndim != 2 or demos.shape[1] != len(rule):
        raise ValueError("embedding penalty inputs are not aligned")
    rule_norm = np.linalg.norm(rule)
    demo_norm = np.linalg.norm(demos, axis=1)
    if rule_norm == 0.0 or np.any(demo_norm == 0.0):
        raise ValueError("embedding penalty does not accept zero vectors")
    cosine = demos @ rule / (demo_norm * rule_norm)
    return 0.05 * max(0.0, float(np.max(cosine)))


def near_far_indices(
    demo_embeddings: np.ndarray, heldout_embeddings: np.ndarray, *, stratum_size: int = 50,
) -> dict:
    demos = np.asarray(demo_embeddings, dtype=float)
    heldout = np.asarray(heldout_embeddings, dtype=float)
    if demos.ndim != 2 or heldout.ndim != 2 or demos.shape[1] != heldout.shape[1]:
        raise ValueError("near/far embeddings are not aligned")
    if len(heldout) < 2 * int(stratum_size):
        raise ValueError("heldout set is too small for disjoint near/far strata")
    demos = demos / np.linalg.norm(demos, axis=1, keepdims=True)
    heldout = heldout / np.linalg.norm(heldout, axis=1, keepdims=True)
    similarity = np.max(heldout @ demos.T, axis=1)
    order = np.argsort(-similarity, kind="stable")
    return {
        "near": sorted(map(int, order[:int(stratum_size)])),
        "far": sorted(map(int, order[-int(stratum_size):])),
        "max_demo_cosine": similarity,
    }

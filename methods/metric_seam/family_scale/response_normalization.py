#!/usr/bin/env python3
"""Normalize raw model text for the strict semantic alignment parser.

This module repairs exactly two benign formatting slips observed in real model
outputs (unfenced JSON with trailing duplicate braces, and singleton clusters).
It NEVER weakens any real validity check (one-per-fleet, exact partition,
lexical-exact lower bound all stay fully enforced by the untouched downstream
parser); it only repairs syntax/formatting noise that has no bearing on the
semantic content of the response. It must NEVER raise: on any failure or
ambiguity, it falls back to returning the original raw string unchanged so the
strict parser is always the final authority on validity.
"""

from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def normalize_alignment_raw(raw: str) -> str:
    """Normalize raw alignment text, falling back to the original on any error."""
    try:
        match = _FENCE_RE.search(raw)
        candidate = match.group(1) if match else raw

        decoder = json.JSONDecoder()
        try:
            value, end_index = decoder.raw_decode(candidate.strip())
        except json.JSONDecodeError:
            return raw

        trailing = candidate.strip()[end_index:]
        if trailing.strip() not in ("", "}", "]"):
            return raw

        if isinstance(value, dict) and set(value.keys()) == {"clusters", "unmatched_unit_ids"}:
            new_clusters = []
            new_unmatched = list(value.get("unmatched_unit_ids", []))
            for cluster in value.get("clusters", []):
                if (
                    isinstance(cluster, dict)
                    and set(cluster.keys()) == {"cluster_id", "unit_ids"}
                    and isinstance(cluster.get("unit_ids"), list)
                    and len(cluster.get("unit_ids")) == 1
                ):
                    new_unmatched.append(cluster["unit_ids"][0])
                else:
                    new_clusters.append(cluster)
            value = {"clusters": new_clusters, "unmatched_unit_ids": new_unmatched}

        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return raw

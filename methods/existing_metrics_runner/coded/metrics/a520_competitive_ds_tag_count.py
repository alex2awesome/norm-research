"""a520: Competitive-programming data-structure / algorithm tag count.

Univariate function of CANDIDATE CODE ALONE.

Counts the number of DISTINCT canonical competitive-programming
algorithm/data-structure categories that the candidate code mentions
(via keyword or canonical-name patterns). Each category is detected once
per code blob.

Categories (10):
  segtree    — segment tree
  fenwick    — Fenwick / BIT
  dsu        — union-find / disjoint set
  trie       — trie / Aho-Corasick
  sparse     — sparse table / RMQ
  bbst       — treap / splay / link-cut tree
  string_alg — KMP / Z / suffix array / Manacher / polynomial hash
  graph_sp   — Dijkstra / Bellman-Ford / Floyd / SPFA
  dp         — explicit dp[] / memo[] / memoization decorator
  pq_set     — std::priority_queue / heapq / std::set / std::map

Return: integer count in [0, 10].

CLASSIFICATION: THIN — deterministic count of keyword categories.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a520"
ASPECT_NAME = "Competitive-DS algorithm tag count"
TIER = 1
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

# Patterns (case-insensitive for name-style tokens; literal for specific
# canonical idioms). Each maps category -> compiled pattern.
CATEGORIES = {
    "segtree":   re.compile(r"segment[_\s]*tree|segtree|seg_tree|"
                            r"\bbuild\s*\(\s*1\s*,", re.IGNORECASE),
    "fenwick":   re.compile(r"fenwick|bit[_\s]*tree|binary[_\s]+indexed[_\s]+tree",
                            re.IGNORECASE),
    "dsu":       re.compile(r"union[\s_]{0,3}find|\bdsu\b|disjoint[\s_]{0,3}set|"
                            r"\bparent\s*\[", re.IGNORECASE),
    "trie":      re.compile(r"\btrie\b|aho[_\s\-]*corasick", re.IGNORECASE),
    "sparse":    re.compile(r"sparse[\s_]{0,5}table|\bRMQ\b", re.IGNORECASE),
    "bbst":      re.compile(r"\btreap\b|\bsplay\b|link[\s_\-]*cut", re.IGNORECASE),
    "string_alg": re.compile(
        r"\bKMP\b|z[\s_]{0,5}function|suffix[\s_]{0,5}array|"
        r"manacher|polynomial[\s_]+hash|rolling[\s_]+hash",
        re.IGNORECASE),
    "graph_sp":  re.compile(r"dijkstra|bellman[\s_\-]*ford|floyd|\bspfa\b|"
                            r"a\*[\s_]*search", re.IGNORECASE),
    "dp":        re.compile(r"\bdp\s*\[|\bmemo\s*\[|memoiz|@lru_cache|@cache|"
                            r"@functools\.lru_cache|@functools\.cache",
                            re.IGNORECASE),
    "pq_set":    re.compile(r"priority_queue|heapq|heappush|heappop|"
                            r"std::set\b|std::map\b|std::multiset\b|"
                            r"std::multimap\b|std::unordered_set|"
                            r"std::unordered_map|\bbisect\b", re.IGNORECASE),
}


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    blob = "\n".join(by_path.values())
    if not blob.strip():
        return None
    n = 0
    for _, pat in CATEGORIES.items():
        if pat.search(blob):
            n += 1
    return float(n)

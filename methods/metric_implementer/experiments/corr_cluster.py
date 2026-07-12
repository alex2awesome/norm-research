#!/usr/bin/env python3
"""Correlation clustering: turn a set of +/- same/different edges into a partition that MINIMIZES
disagreements (split +edges + merged -edges). This is the principled replacement for (a) the brittle
union-find/min_votes reconcile (which lost coverage) and (b) the tau=0.92 HAC that undermerged.

ANTI-DRIFT property (the user's concern): unlike single-linkage/union-find (where A~B, B~C chains
A,B,C together even if A,C is a -edge), local-search correlation clustering places a node in the
cluster where it has the MOST agreement — so a -edge between A and C can keep them apart despite a
+ path through B. Drift chains get cut when the - evidence outweighs.

Also reusable as the R1/R2 primitive: cluster L0 representatives (or R1 representatives) by their
+/- pairwise edges into the next level.

Algorithm: Lloyd-style local search. Each node starts in its own cluster; repeatedly move each node
to the cluster (incl. a fresh singleton) that minimizes its disagreements with that cluster's members,
until stable. Deterministic (fixed node order + seed for tie-breaks).
"""
from __future__ import annotations
import json, os, sys, random
from collections import defaultdict
import numpy as np


def build_edges_from_sims(keys, sims, theta_hi, theta_lo):
    """+ edge if sim>=theta_hi (same), - edge if sim<theta_lo (different); in between = neutral (no edge,
    to be filled by GLM-4.7 in the hybrid). Returns plus:set[frozenset], minus:set[frozenset]."""
    n = len(keys)
    plus, minus = set(), set()
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s >= theta_hi:
                plus.add(frozenset((i, j)))
            elif s < theta_lo:
                minus.add(frozenset((i, j)))
    return plus, minus


def best_threshold(rows, lo=0.3, hi=0.95, step=0.01):
    """Sweep theta; treat sim>=theta as predicted-same. Return theta maximizing F1 vs gold (label==2)."""
    best = (0.0, 0.0)
    for th in np.arange(lo, hi, step):
        tp = fp = fn = 0
        for sim, g in rows:
            pred = sim >= th
            if pred and g == 2:
                tp += 1
            elif pred and g != 2:
                fp += 1
            elif (not pred) and g == 2:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if f1 > best[1]:
            best = (float(th), f1)
    return best[0]


def correlation_cluster(nodes, plus, minus, max_iter=25, seed=7):
    """Local-search correlation clustering, O(degree) per node per iter. nodes: list of ids;
    plus/minus: sets of frozenset({a,b}). For each node we pick the cluster (or a singleton) that
    minimizes its disagreements: cost(cluster c) = (#-edges from v into c) + (#+edges from v NOT into c);
    cost(singleton) = total #+edges. This is the anti-drift step: a -edge to c raises cost(c), so a
    chain A-B-C with A,C negative keeps A,C apart despite the + path through B."""
    adj = defaultdict(dict)                        # node -> {neighbor: +1/-1}
    for e in plus:
        a, b = tuple(e); adj[a][b] = 1; adj[b][a] = 1
    for e in minus:
        a, b = tuple(e); adj[a][b] = -1; adj[b][a] = -1
    label = {v: v for v in nodes}
    for it in range(max_iter):
        moved = False
        for v in nodes:
            plus_c = defaultdict(int); minus_c = defaultdict(int); total_plus = 0
            for nb, s in adj[v].items():
                c = label[nb]
                if s == 1:
                    plus_c[c] += 1; total_plus += 1
                else:
                    minus_c[c] += 1
            cur = label[v]
            cur_cost = minus_c.get(cur, 0) + (total_plus - plus_c.get(cur, 0))
            best_c, best_cost = None, total_plus      # None = singleton (cost = all +edges split)
            for c in set(plus_c) | set(minus_c):
                cost = minus_c[c] + (total_plus - plus_c[c])
                if cost < best_cost:
                    best_cost, best_c = cost, c
            target = best_c if best_c is not None else v
            if best_cost < cur_cost and target != cur:
                label[v] = target; moved = True
        if not moved:
            break
    final = {}; nxt = 0
    for v in nodes:
        c = label[v]
        if c not in final:
            final[c] = nxt; nxt += 1
    return {v: final[label[v]] for v in nodes}


def score_partition(part, pairs, labels):
    """part: {key: cid}; recall/precision vs arbiter labels on the given pairs."""
    s2 = s0 = tog2 = sep0 = 0
    for pid, p in pairs.items():
        g = labels.get(pid)
        ka, kb = p["key_a"], p["key_b"]
        if g is None or ka not in part or kb not in part:
            continue
        same = part[ka] == part[kb]
        if g == 2:
            s2 += 1; tog2 += same
        elif g == 0:
            s0 += 1; sep0 += (not same)
    rec = tog2 / s2 if s2 else None
    pre = sep0 / s0 if s0 else None
    return rec, pre, s2, s0

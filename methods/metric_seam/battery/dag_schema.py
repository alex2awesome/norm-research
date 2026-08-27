"""WS4: typed-DAG metric programs — schema, executor, level-match checker (runbook WS4).

A DAG program makes seam POSITION structural: instead of one scalar hybrid where judgment
hides anywhere, every node declares WHAT it is and the trace shows WHERE judgment lives.

Node fields:
  id        unique string
  ntype     'R' retrieve | 'T' transform | 'A' assess
  impl      'C' code | 'L' LLM-field (value arrives via `extracted`, None during iteration)
  op_class  'evidence' (cites the evidence-aware ceiling, e.g. patents M̄(x,Z)) |
            'computation' (cites the doc-only ceiling)
  level     abstraction level of the node's OUTPUT: 0 raw-text < 1 span < 2 feature < 3 verdict
  needs     list of upstream node ids (edges)
  fn        callable(ctxdict) -> value; ctxdict has 'text', 'extracted', 'ops' plus one key
            per upstream node id holding that node's output. L-nodes read their field from
            extracted and may post-process; their fn must tolerate None.

Program = {"nodes": [node...], "out": <id of the final A node>}.

LEVEL-MATCH RULE (per-edge, checked statically): an edge u->v requires level(u) <= level(v)
— information may only move up the abstraction lattice; a verdict feeding a span-extractor
is a level inversion (the WS4 analogue of judging evidence the judge can't see).

SEAM READOUT per program: seam_frontier(prog) = the set of minimal L-nodes under DAG order
(judgment entry points) + depth stats. Executor records a full per-node trace per item;
node-level ablation (replace node output with its train-median) gives per-node marginals
certified against the op_class-matched ceiling.

Gate machinery unchanged: the final score column feeds the same G1 held-out bootstrap.
"""
from collections import deque


LEVELS = {0: "raw-text", 1: "span", 2: "feature", 3: "verdict"}
REQUIRED = ("id", "ntype", "impl", "op_class", "level", "needs", "fn")


def validate(prog):
    """Static checks: schema completeness, DAG-ness, per-edge level match. -> list of errors."""
    errs = []
    nodes = {n["id"]: n for n in prog["nodes"]}
    if len(nodes) != len(prog["nodes"]):
        errs.append("duplicate node ids")
    if prog["out"] not in nodes:
        errs.append(f"out node {prog['out']} missing")
    for n in prog["nodes"]:
        for k in REQUIRED:
            if k not in n:
                errs.append(f"{n.get('id','?')}: missing {k}")
        if n.get("ntype") not in ("R", "T", "A"):
            errs.append(f"{n['id']}: bad ntype")
        if n.get("impl") not in ("C", "L"):
            errs.append(f"{n['id']}: bad impl")
        if n.get("op_class") not in ("evidence", "computation"):
            errs.append(f"{n['id']}: bad op_class")
        for u in n.get("needs", []):
            if u not in nodes:
                errs.append(f"{n['id']}: unknown dep {u}")
            elif nodes[u]["level"] > n["level"]:
                errs.append(f"LEVEL INVERSION {u}(L{nodes[u]['level']}) -> "
                            f"{n['id']}(L{n['level']})")
    # DAG check (Kahn)
    indeg = {i: len(n["needs"]) for i, n in nodes.items()}
    q = deque([i for i, d in indeg.items() if d == 0])
    seen = 0
    kids = {i: [] for i in nodes}
    for n in prog["nodes"]:
        for u in n["needs"]:
            if u in kids:
                kids[u].append(n["id"])
    while q:
        seen += 1
        for c in kids[q.popleft()]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if seen != len(nodes):
        errs.append("cycle detected")
    return errs


def topo_order(prog):
    nodes = {n["id"]: n for n in prog["nodes"]}
    order, done = [], set()

    def visit(i):
        if i in done:
            return
        for u in nodes[i]["needs"]:
            visit(u)
        done.add(i)
        order.append(i)

    for i in nodes:
        visit(i)
    return [nodes[i] for i in order]


def run(prog, text, extracted, ops, trace=None):
    """Execute one item. trace (optional dict) receives every node's output."""
    ctx = {"text": text, "extracted": extracted or {}, "ops": ops}
    for n in topo_order(prog):
        try:
            ctx[n["id"]] = n["fn"](ctx)
        except Exception:
            ctx[n["id"]] = None
        if trace is not None:
            trace[n["id"]] = ctx[n["id"]]
    return ctx[prog["out"]]


def seam_frontier(prog):
    """Minimal L-nodes under DAG order = where judgment ENTERS the program."""
    nodes = {n["id"]: n for n in prog["nodes"]}
    anc = {}

    def ancestors(i):
        if i not in anc:
            s = set()
            for u in nodes[i]["needs"]:
                s |= {u} | ancestors(u)
            anc[i] = s
        return anc[i]

    lnodes = [i for i, n in nodes.items() if n["impl"] == "L"]
    frontier = [i for i in lnodes if not any(j in ancestors(i) for j in lnodes)]
    return {"frontier": sorted(frontier), "n_L": len(lnodes),
            "n_nodes": len(nodes),
            "frontier_levels": sorted(nodes[i]["level"] for i in frontier)}


def score_fn(prog):
    """Adapter: DAG program -> the fleet's score(text, extracted, ops) signature."""
    def score(text, extracted, ops):
        return run(prog, text, extracted, ops)
    return score

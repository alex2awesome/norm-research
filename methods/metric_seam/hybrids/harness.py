"""Hybrid-channel evaluation harness — seam pilot v1.

Judge target = mean of pass1/pass2 Gemma scores (items where both parse & numeric).
Split: 150 train / 100 test by datapoint_id (seed 7).
Gate (per proposal §4.2, pilot form):
  G1 faithfulness:    Spearman rho_test vs judge >= max(baseline_rho_test + 0.10, 0.60)
  G2 non-inferiority: evaluated on the SAME held-out items as baseline
  G3 CF (a86 only):   hybrid must not raise score on quote-injected CF items more than the
                      judge does (margin 0.05 on [0,1] scale)
"""
import importlib.util, json, math, pathlib, random, signal, statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/v1"
FIELDS_DIR = OUT / "llm_fields"

def _alarm(sig, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)


def load_judge():
    """aspect -> dpid -> combined 0-1 score (mean of available numeric passes)."""
    p1, p2 = {}, {}
    for line in open(OUT / "results_v1.jsonl"):
        r = json.loads(line)
        if isinstance(r["score"], int):
            d = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
            if d is not None:
                d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    combined = {}
    for aid in set(p1) | set(p2):
        for dpid in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [d[aid][dpid] for d in (p1, p2) if dpid in d.get(aid, {})]
            combined.setdefault(aid, {})[dpid] = sum(vals) / len(vals) / 10.0
    return combined, p1, p2


def load_scope(thresh=7):
    sc = {}
    for line in open(OUT / "results_v1.jsonl"):
        r = json.loads(line)
        if r["channel"] == "scope" and isinstance(r["score"], int):
            sc[r["datapoint_id"]] = r["score"]
    return {d for d, s in sc.items() if s >= thresh}, sc


def split_ids():
    items = json.load(open(OUT / "items_v1.json"))
    ids = sorted(x["datapoint_id"] for x in items)
    random.Random(7).shuffle(ids)
    return set(ids[:150]), set(ids[150:])


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 10:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def load_hybrid(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_fields(aid):
    """dpid -> {field: answer}; empty if no fields extracted yet."""
    out = {}
    if not FIELDS_DIR.exists():
        return out
    for f in FIELDS_DIR.glob(f"{aid}__*.json"):
        field = f.stem.split("__", 1)[1]
        for dpid, ans in json.load(open(f)).items():
            out.setdefault(dpid, {})[field] = ans
    return out


def run_hybrid(mod, texts, fields, ops):
    col = {}
    for dpid, t in texts.items():
        try:
            signal.alarm(15)
            col[dpid] = float(mod.score(t, fields.get(dpid, {}), ops))
        except Exception:
            col[dpid] = None
        finally:
            signal.alarm(0)
    return col


def evaluate(aid, hybrid_path, ops, scope_only=False):
    judge, _, _ = load_judge()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items_v1.json"))}
    train, test = split_ids()
    in_scope, _ = load_scope()
    mod = load_hybrid(hybrid_path)
    fields = load_fields(aid)
    scores = run_hybrid(mod, items, fields, ops)
    res = {}
    for name, idset in [("train", train), ("test", test)]:
        sel = [d for d in idset if d in judge.get(aid, {}) and scores.get(d) is not None]
        if scope_only:
            sel = [d for d in sel if d in in_scope]
        xs = [scores[d] for d in sel]
        ys = [judge[aid][d] for d in sel]
        res[name] = {"n": len(sel), "rho": round(spearman(xs, ys), 3)}
    res["scores"] = scores
    return res

"""EXP-VOICE-NOUS-1 — LEG B GENERATION (prereg: notes/2026-08-15__exp-voice-nous-prereg.md;
frozen materials: outputs/exp_voice_nous/legB_materials_v1.json sha256 prefix 2d4a0f6a37535f1c).

12 personas (4 D1 rule-defined / 4 D2 style-card / 4 D3 seed-exemplar) x 20 shared topics x 2
drafts = 480 pieces, 550-750 words, ONE generator for all personas (gpt-oss-120b, temp 0.9,
harmony final-channel only — the authoring-leak fix). D3 seed authors resolved by the materials
collision policy against the Leg A slate: Teddy Wayne, Suzanne Yeagley, Ben Greenman,
Kevin Dolgin (6 seed excerpts each, seed 0, boilerplate-stripped, 300-word cap).

Modes:
  build       (CPU) : 480 generation prompts (12 smoke rows FIRST) -> <out>/legb_gen_prompts.jsonl
  run         (sk3) : chunked+resumable generation                 -> <out>/legb_gen_raw.jsonl
                      (--n 12 = smoke subset)
  check       (CPU) : D1 checker pass rates; --emit-regen writes forceful retry prompts for
                      failing D1 texts of personas under 80% pass  -> <out>/legb_regen_prompts.jsonl
  regen       (sk3) : generate the regen prompts                   -> <out>/legb_regen_raw.jsonl
  pack        (CPU) : final artifact + manifest -> outputs/exp_voice_nous/legB_texts_v1.jsonl.gz

Checker notes (frozen instrument, declared implementation details): regex checkers are compiled
with re.S (the frozen '^...$' full-text patterns otherwise fail on any multiline text, which
cannot be the intent); SPECIAL checkers implemented in python below; paragraphs = blank-line
separated blocks.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys

from .exp_voice_smoke import _excerpt, _load_corpus, _rng

SEED = 0
MATERIALS = "outputs/exp_voice_nous/legB_materials_v1.json"
MATERIALS_SHA = "2d4a0f6a37535f1c"
GEN_MODEL = ("gpt-oss-120b", "openai/gpt-oss-120b")
TEMP = 0.9
WORD_BAND = (550, 750)
REGEN_THRESHOLD = 0.80


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _harmony_final(text):
    for marker in ("<|channel|>final<|message|>", "assistantfinal", "\nfinal\n"):
        if marker in text:
            return text.split(marker)[-1].split("<|")[0].strip()
    return ""


def _load_materials(root):
    raw = open(os.path.join(root, MATERIALS), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    if sha != MATERIALS_SHA:
        sys.exit(f"[legb] materials sha mismatch: {sha} != {MATERIALS_SHA}")
    return json.loads(raw)


# ---------------------------------------------------------------- D1 checkers ----------------
# v2 scoring (2026-08-15): the generation prompt explicitly allowed "a title line", so all
# checkers run on the TITLE-STRIPPED body. v1 scores (title penalized) live in the original
# artifact rows; the rescore artifact records both.
def _strip_title(text):
    lines = text.strip().splitlines()
    if not lines:
        return text.strip()
    first = lines[0].strip()
    words = re.findall(r"[A-Za-z0-9'\u2019-]+", first)
    is_heading = first.startswith("#") or (first.startswith("**") and first.endswith("**"))
    if is_heading or (len(words) <= 12 and not re.search(r"[.!?]$", first)
                      and not first.startswith(("Dear ", "Q:", "A:", "1.", "1)"))):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _paras(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _final_sentence_word_count(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sents:
        return 0
    return len(re.findall(r"[A-Za-z0-9'’-]+", sents[-1]))


_SPECIAL = {
    "SPECIAL:final_sentence_word_count==3": lambda t: _final_sentence_word_count(t) == 3,
    "SPECIAL:each_para_contains_you":
        lambda t: all(re.search(r"(?i)\byou\b", p) for p in _paras(t)),
    "SPECIAL:each_para_one_paren":
        lambda t: all(p.count("(") == 1 and p.count(")") == 1 for p in _paras(t)),
    "SPECIAL:count('P.S.')==1": lambda t: t.count("P.S.") == 1,
    "SPECIAL:count_lines_starting('Q:')>=8":
        lambda t: sum(l.strip().startswith("Q:") for l in t.splitlines()) >= 8,
    "SPECIAL:each_A_le_25_words":
        lambda t: all(len(l.strip()[2:].split()) <= 25
                      for l in t.splitlines() if l.strip().startswith("A:")),
}


def run_checkers(checkers, text):
    text = _strip_title(text)   # v2: title line exempt (generation prompt allowed one)
    out = {}
    for name, spec in checkers.items():
        if spec.startswith("SPECIAL:"):
            out[name] = bool(_SPECIAL[spec](text))
        else:
            out[name] = bool(re.search(spec, text, re.S))
    return out


# ---------------------------------------------------------------- build ----------------------
def _personas(mat, root):
    """-> list of {persona, grade, block} in fixed order (D1 x4, D2 x4, D3 x4)."""
    ps = []
    for spec in mat["grades"]["D1"]:
        block = mat["generation"]["persona_blocks"]["D1"].format(
            rules="\n- ".join(spec["rules"]))
        ps.append({"persona": spec["id"], "grade": "D1", "block": block,
                   "checkers": spec["checkers"], "rules": spec["rules"]})
    for spec in mat["grades"]["D2"]:
        block = mat["generation"]["persona_blocks"]["D2"].format(card=spec["card"])
        ps.append({"persona": spec["id"], "grade": "D2", "block": block})
    slate = json.load(open(os.path.join(
        root, "outputs/exp_voice_nous/legA_fleet_v1.json")))["manifest"]["slate"]
    seeds_auth = [a for a in mat["grades"]["D3"]["seed_candidates"] if a not in slate][:4]
    pieces = _load_corpus(root)
    by_author = {}
    for p in pieces:
        by_author.setdefault(p["author"], []).append(p)
    for v in by_author.values():
        v.sort(key=lambda p: p["_id"])
    k = mat["grades"]["D3"]["seeds_per_persona"]
    for au in seeds_auth:
        picks = _rng("voice-legb-seeds", SEED, au).sample(by_author[au], k)
        exc = "\n\n".join(f"[Piece {i+1}]\n{_excerpt(p['text'])}" for i, p in enumerate(picks))
        block = mat["generation"]["persona_blocks"]["D3"].format(seed_excerpts=exc)
        ps.append({"persona": f"D3-{_slug(au)}", "grade": "D3", "block": block,
                   "seed_author": au, "seed_ids": [p["_id"] for p in picks]})
    return ps, seeds_auth


def build(a):
    mat = _load_materials(a.root)
    personas, seeds_auth = _personas(mat, a.root)
    tpl = mat["generation"]["prompt_template"]
    topics = mat["topics"]
    smoke_rows, rest_rows = [], []
    for p in personas:
        order = list(range(len(topics)))
        _rng("voice-legb-topics", SEED, p["persona"]).shuffle(order)
        for rank, ti in enumerate(order):
            for draft in range(mat["drafts_per_topic"]):
                key = f"{p['persona']}|t{ti}|d{draft}"
                row = {"key": key, "persona": p["persona"], "grade": p["grade"],
                       "topic_idx": ti, "topic": topics[ti], "draft": draft,
                       "gen_seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16),
                       "prompt": tpl.format(persona_block=p["block"], topic=topics[ti])}
                (smoke_rows if rank == 0 and draft == 0 else rest_rows).append(row)
    rows = smoke_rows + rest_rows                      # smoke dozen first -> run --n 12
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "legb_gen_prompts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"seed": SEED, "materials_sha": MATERIALS_SHA, "generator": GEN_MODEL,
               "temperature": TEMP, "d3_seed_authors": seeds_auth,
               "personas": [{k: v for k, v in p.items() if k != "block"} |
                            {"block_words": len(p["block"].split())} for p in personas],
               "n_prompts": len(rows)},
              open(os.path.join(a.out, "legb_state.json"), "w"), indent=1)
    print(f"[build] {len(rows)} gen prompts ({len(smoke_rows)} smoke first) -> {a.out}")


# ---------------------------------------------------------------- run / regen (sk3) ----------
def _generate(a, src, dst):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    rows = [json.loads(l) for l in open(os.path.join(a.out, src))]
    if a.n:
        rows = rows[: a.n]
    df = os.path.join(a.out, dst)
    done = {json.loads(l)["key"] for l in open(df)} if os.path.exists(df) else set()
    todo = [r for r in rows if r["key"] not in done]
    print(f"[gen] {len(todo)} to generate ({len(done)} already done)")
    if not todo:
        return
    cfg = cfgmod.ImplementerConfig()
    ex = make_judge_backend(GEN_MODEL[1], cfg, temperature=TEMP)

    def valid(t):
        # retry pressure toward the 550-750 target (smoke showed D2/D3 overshooting to ~1000);
        # after max_retries the last draft is kept regardless — band compliance is REPORTED.
        w = len(_harmony_final(t).split())
        return 400 <= w <= 850

    CH = 96
    with open(df, "a") as f:
        for lo in range(0, len(todo), CH):
            chunk = todo[lo:lo + CH]
            outs = ex.generate_batch([r["prompt"] for r in chunk], max_tokens=5000,
                                     validate=valid, seed=[r["gen_seed"] for r in chunk])
            for r, o in zip(chunk, outs):
                text = _harmony_final(o)
                f.write(json.dumps({"key": r["key"], "persona": r["persona"],
                                    "grade": r["grade"], "topic_idx": r["topic_idx"],
                                    "topic": r["topic"], "draft": r["draft"],
                                    "n_words": len(text.split()), "text": text}) + "\n")
            f.flush()
            print(f"[gen] {min(lo+CH, len(todo))}/{len(todo)}", flush=True)
    print(f"[gen] DONE -> {df}")


def run(a):
    _generate(a, "legb_gen_prompts.jsonl", "legb_gen_raw.jsonl")


def regen(a):
    _generate(a, "legb_regen_prompts.jsonl", "legb_regen_raw.jsonl")


# ---------------------------------------------------------------- check ----------------------
def _d1_specs(mat):
    return {s["id"]: s for s in mat["grades"]["D1"]}


def check(a):
    mat = _load_materials(a.root)
    specs = _d1_specs(mat)
    rows = [json.loads(l) for l in open(os.path.join(a.out, "legb_gen_raw.jsonl"))]
    stats, fails = {}, []
    for pid, spec in specs.items():
        mine = [r for r in rows if r["persona"] == pid]
        n_pass = 0
        for r in mine:
            res = run_checkers(spec["checkers"], r["text"])
            if all(res.values()):
                n_pass += 1
            else:
                fails.append((r, res))
        stats[pid] = {"n": len(mine), "pass": n_pass,
                      "rate": n_pass / len(mine) if mine else None}
    for pid, s in stats.items():
        print(f"[check] {pid:16s} pass {s['pass']}/{s['n']} = "
              f"{s['rate']:.3f}" if s["n"] else f"[check] {pid} EMPTY")
    if a.emit_regen:
        gen_rows = {json.loads(l)["key"]: json.loads(l)
                    for l in open(os.path.join(a.out, "legb_gen_prompts.jsonl"))}
        out = []
        for r, res in fails:
            if stats[r["persona"]]["rate"] >= REGEN_THRESHOLD:
                continue                                # only personas under threshold
            g = gen_rows[r["key"]]
            rules = "\n- ".join(specs[r["persona"]]["rules"])
            forceful = (g["prompt"] + "\n\nIMPORTANT: A previous draft violated the rules. "
                        "You MUST obey EVERY rule below EXACTLY — check each one before "
                        "finishing:\n- " + rules)
            out.append(g | {"prompt": forceful,
                            "gen_seed": g["gen_seed"] + 7_777_777,
                            "failed_checkers": [k for k, v in res.items() if not v]})
        with open(os.path.join(a.out, "legb_regen_prompts.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"[check] {len(out)} regen prompts -> legb_regen_prompts.jsonl")


# ---------------------------------------------------------------- pack -----------------------
def pack(a):
    mat = _load_materials(a.root)
    specs = _d1_specs(mat)
    state = json.load(open(os.path.join(a.out, "legb_state.json")))
    rows = {r["key"]: r for r in
            (json.loads(l) for l in open(os.path.join(a.out, "legb_gen_raw.jsonl")))}
    rf = os.path.join(a.out, "legb_regen_raw.jsonl")
    regens = ({r["key"]: r for r in (json.loads(l) for l in open(rf))}
              if os.path.exists(rf) else {})
    n_swap = 0
    final = []
    for key, r in rows.items():
        chosen, src = r, "original"
        if r["grade"] == "D1":
            res0 = run_checkers(specs[r["persona"]]["checkers"], r["text"])
            if key in regens:
                res1 = run_checkers(specs[r["persona"]]["checkers"], regens[key]["text"])
                if sum(res1.values()) > sum(res0.values()):
                    chosen, src = regens[key], "regen"
                    n_swap += 1
            cres = run_checkers(specs[chosen["persona"]]["checkers"], chosen["text"])
        else:
            cres = None
        final.append({"persona": chosen["persona"], "grade": chosen["grade"],
                      "topic_idx": chosen["topic_idx"], "topic": chosen["topic"],
                      "draft": chosen["draft"], "source": src,
                      "n_words": len(chosen["text"].split()),
                      "checker_results": cres, "text": chosen["text"]})
    dst = os.path.join(a.root, "outputs/exp_voice_nous/legB_texts_v1.jsonl.gz")
    with gzip.open(dst, "wt") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    d1 = [r for r in final if r["grade"] == "D1"]
    d1_pass = {}
    for pid in specs:
        mine = [r for r in d1 if r["persona"] == pid]
        d1_pass[pid] = sum(all(r["checker_results"].values()) for r in mine) / len(mine)
    man = {"experiment": "EXP-VOICE-NOUS-1", "stage": "Leg B generation",
           "materials_sha": MATERIALS_SHA, "generator": GEN_MODEL, "temperature": TEMP,
           "n_texts": len(final), "regen_swapped_in": n_swap,
           "d3_seed_authors": state["d3_seed_authors"], "personas": state["personas"],
           "word_band_target": WORD_BAND,
           "in_band_rate": sum(WORD_BAND[0] <= r["n_words"] <= WORD_BAND[1]
                               for r in final) / len(final),
           "d1_final_pass_rates": d1_pass,
           "notes": ["harmony final-channel extraction (assistantfinal), CoT never stored",
                     "regex checkers compiled with re.S; SPECIAL checkers in "
                     "exp_voice_legb.py; paragraphs = blank-line blocks"]}
    json.dump(man, open(os.path.join(
        a.root, "outputs/exp_voice_nous/legB_texts_v1_manifest.json"), "w"), indent=1)
    print(f"[pack] {len(final)} texts -> {dst} (regen swapped: {n_swap})")
    print(json.dumps(man["d1_final_pass_rates"], indent=1))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "run", "regen", "check", "pack"])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=0, help="run only first N prompts (smoke=12)")
    p.add_argument("--emit-regen", action="store_true")
    a = p.parse_args(argv)
    {"build": build, "run": run, "regen": regen, "check": check, "pack": pack}[a.mode](a)


if __name__ == "__main__":                                   # spawn-safety (sk3 vLLM rule)
    main()

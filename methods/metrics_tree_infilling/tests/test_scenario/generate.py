"""Generate the creature-dossier corpus.

Writes:
  corpus.csv      — id, text, judgement   (the model's view: text + the elders' verdict)
  answer_key.csv  — id + every ground-truth attribute, region, logit, p(kept)

Usage:
  PYTHONPATH=methods python -m metrics_tree_infilling.tests.test_scenario.generate \
      --n 1500 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import world

HERE = Path(__file__).resolve().parent


def build_corpus(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, key = [], []
    for i in range(n):
        attrs = world.sample_attrs(rng)
        name = world._name(rng)
        text = world.build_text(attrs, rng, name=name)
        logit = world.label_logit(attrs)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = int(rng.random() < p)
        rows.append({"id": i, "text": text, "judgement": y})
        key.append({"id": i, "name": name, **attrs,
                    "region": attrs["habitat"], "logit": round(logit, 4), "p_kept": round(p, 4)})
    return pd.DataFrame(rows), pd.DataFrame(key)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args(argv)

    corpus, key = build_corpus(args.n, args.seed)
    out = Path(args.out_dir)
    corpus.to_csv(out / "corpus.csv", index=False)
    key.to_csv(out / "answer_key.csv", index=False)

    print(f"Wrote {len(corpus)} items to {out}")
    print(f"  overall kept-rate: {corpus['judgement'].mean():.3f}")
    for region in ("grove", "marsh", "cavern"):
        m = (key["region"] == region).values
        print(f"  {region:6s}: n={m.sum():4d} ({m.mean():.0%})  kept-rate={corpus.loc[m,'judgement'].mean():.3f}")
    # ground-truth signal of each tacit norm WITHIN its region
    marsh = (key["region"] == "marsh").values
    mel = corpus.loc[key.index[marsh & (key['song'] == 'melodious')], 'judgement'].mean()
    har = corpus.loc[key.index[marsh & (key['song'] == 'harsh')], 'judgement'].mean()
    print(f"  marsh  song: melodious kept={mel:.2f} vs harsh kept={har:.2f}")
    cav = (key["region"] == "cavern").values
    lum = corpus.loc[key.index[cav & (key['glow'] == 'luminous')], 'judgement'].mean()
    dim = corpus.loc[key.index[cav & (key['glow'] == 'dim')], 'judgement'].mean()
    print(f"  cavern glow: luminous kept={lum:.2f} vs dim kept={dim:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

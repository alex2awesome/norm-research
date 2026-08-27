import json, os, sys
import numpy as np
B='/lfs/skampere3/0/alexspan'; O=f'{B}/outputs/osl_multi'
sys.path.insert(0, f'{B}/norm-research')
os.environ['OSL_PROBES_FILE'] = f'{O}/humor_long_probes.jsonl'
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.experiments.run_real_test import _load_texts
frz = json.load(open(f'{O}/freeze_zxa_humor_v1.json'))
cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), frz['meta']['task'])
texts, _ = _load_texts(frz['meta']['task'], 360, cfg)
rows = [json.loads(l) for l in open(f'{O}/humor_long_probes.jsonl')]
match = sum(texts[60+i] == rows[i]['text'] for i in range(300))
print(f'ALIGNMENT: {match}/300 exact matches at offset 60')
if match < 300:
    # where do file rows actually sit?
    pos = {t: i for i, t in enumerate(texts)}
    hits = [pos.get(rows[i]['text'], -1) for i in range(5)]
    print('first 5 file rows found at text indices:', hits)
    print('texts[60][:70] =', repr(texts[60][:70]))
    print('file[0][:70]  =', repr(rows[0]['text'][:70]))
# yes-rate profile: is hermes on a floor?
def load(tag, ex):
    z = np.load(f'{O}/mbar_zxa{tag}_humor_{ex}.npz', allow_pickle=True)
    return {str(n): r for n, r in zip(z['names'], z['m_bar'].astype(float))}
H = load('LP', 'hermes405b'); L = load('LP', 'llama70b'); Q = load('LP', 'qwen25-72b')
ys_h = [np.nanmean(v) for n, v in H.items()]
ys_l = [np.nanmean(v) for n, v in L.items() if n.endswith('||name')]
ys_q = [np.nanmean(v) for n, v in Q.items() if n.endswith('||name')]
print(f'name-arm yes-rate median (IQR): hermes {np.median(ys_h):.2f} '
      f'({np.percentile(ys_h,25):.2f}-{np.percentile(ys_h,75):.2f}), '
      f'llama70b {np.median(ys_l):.2f}, qwen72 {np.median(ys_q):.2f}')
# raw agreement + lag scan on an anchor metric
anchors = [n for n in H if 'ANCHOR' in str(np.load(f"{O}/mbar_zxaLP_humor_hermes405b.npz", allow_pickle=True)['kinds'][list(H).index(n)])]
n0 = anchors[0] if anchors else list(H)[0]
a = H[n0]; b = L[n0]
best = []
for lag in range(-3, 4):
    aa = np.roll(a, lag)
    m = np.isfinite(aa) & np.isfinite(b)
    best.append((lag, float(((aa[m] > .5) == (b[m] > .5)).mean())))
print(f'lag-scan agreement on {n0[:40]!r}:', best)

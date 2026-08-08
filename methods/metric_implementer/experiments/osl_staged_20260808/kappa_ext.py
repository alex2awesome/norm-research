import numpy as np
O='/lfs/skampere3/0/alexspan/outputs/osl_multi'
def load(tag,ex):
    z=np.load(f'{O}/mbar_zxa{tag}_humor_{ex}.npz',allow_pickle=True)
    names=[str(n) for n in z['names']]; kinds=[str(k) for k in z['kinds']]
    return dict(zip(names,[r for r in z['m_bar'].astype(float)])), dict(zip(names,kinds))
def kap(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<30: return np.nan
    x=(a[m]>0.5).astype(int); y=(b[m]>0.5).astype(int)
    po=(x==y).mean(); pe=x.mean()*y.mean()+(1-x.mean())*(1-y.mean())
    return (po-pe)/(1-pe) if pe<1 else np.nan
V1={};LP={}
for ex in ['llama70b','qwen25-72b','qwen25-32b']:
    V1[ex]=load('',ex); LP[ex]=load('LP',ex)
LP['hermes4-405b']=load('LP','hermes405b')
bases=sorted(set(n.split('||')[0] for n in V1['llama70b'][0]))
KINDS=['TACIT-CANDIDATE','DIALECT-SUSPECT','REACHES-ANCHOR','PLANTED']
def table(pairs,D,label):
    print(f'--- {label} (median kappa, name arm)')
    for e1,e2 in pairs:
        if e1 not in D or e2 not in D: continue
        cells=[]
        for kind in KINDS:
            ks=[]
            for b in bases:
                n=f'{b}||name'
                if n in D[e1][0] and n in D[e2][0] and D[e1][1][n].startswith(kind):
                    ks.append(kap(D[e1][0][n],D[e2][0][n]))
            ks=[k for k in ks if np.isfinite(k)]
            cells.append(f'{kind.split("-")[0][:6]:6s}={np.median(ks):+.2f}(n={len(ks)})' if ks else f'{kind[:6]}=na')
        print(f'  {e1[:12]:13s}x {e2[:12]:13s} '+' '.join(cells))
P=[('llama70b','qwen25-72b'),('llama70b','qwen25-32b'),('qwen25-72b','qwen25-32b')]
table(P,V1,'V1 81-char probes')
table(P,LP,'LP 1018-char probes')
table([('hermes4-405b','llama70b'),('hermes4-405b','qwen25-72b'),('hermes4-405b','qwen25-32b')],LP,'hermes-4-405B (OR) x locals, LP only')
print('--- hermes-4 LP yes-rates on the v1-floor metrics')
for n,v in LP['hermes4-405b'][0].items():
    if any(n.startswith(s) for s in ('Brand identity','PLANTED-length-long','PLANTED-digit')):
        print(f'  {n[:52]:54s} yes={np.nanmean(v):.2f} nan={np.mean(~np.isfinite(v)):.2f}')
print('--- qwen25-72b const rows (arm-collapse check)')
for tag,D in (('v1',V1),('LP',LP)):
    vs=D['qwen25-72b'][0]
    c=sum(1 for v in vs.values() if np.isfinite(v).sum()>=30 and np.nanstd(v)==0)
    print(f'  {tag}: {c}/{len(vs)} const')

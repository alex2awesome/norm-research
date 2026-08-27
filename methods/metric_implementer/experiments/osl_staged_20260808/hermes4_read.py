import json
import numpy as np
O='/lfs/skampere3/0/alexspan/outputs/osl_multi'
def load(path):
    z=np.load(path,allow_pickle=True)
    return {str(n):r for n,r in zip(z['names'],z['m_bar'].astype(float))}, \
           {str(n):str(k) for n,k in zip(z['names'],z['kinds'])}
def kap(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<30: return np.nan
    x=(a[m]>0.5).astype(int); y=(b[m]>0.5).astype(int)
    po=(x==y).mean(); pe=x.mean()*y.mean()+(1-x.mean())*(1-y.mean())
    return (po-pe)/(1-pe) if pe<1 else np.nan
H_v1=load(f'{O}/mbar_zxaV1or_humor_hermes4-405b.npz')
H_lp=load(f'{O}/mbar_zxaLP_humor_hermes4-405b.npz')
loc_v1={e:load(f'{O}/mbar_zxa_humor_{e}.npz') for e in ['llama70b','qwen25-72b','qwen25-32b']}
loc_lp={e:load(f'{O}/mbar_zxaLP_humor_{e}.npz') for e in ['llama70b','qwen25-72b','qwen25-32b']}
KINDS=['TACIT-CANDIDATE','DIALECT-SUSPECT','REACHES-ANCHOR','PLANTED']
print('=== battery ===')
print(json.load(open('/lfs/skampere3/0/alexspan/outputs/osl/hermes4-405b.json'))['battery'])
print('=== hermes-4 kappa vs locals, per kind (name arm): v1 -> LP ===')
for e in ['llama70b','qwen25-72b','qwen25-32b']:
    cells=[]
    for kind in KINDS:
        k1=[];k2=[]
        for n in H_v1[0]:
            if not H_v1[1][n].startswith(kind): continue
            if n in loc_v1[e][0]: k1.append(kap(H_v1[0][n],loc_v1[e][0][n]))
            if n in loc_lp[e][0] and n in H_lp[0]: k2.append(kap(H_lp[0][n],loc_lp[e][0][n]))
        k1=[k for k in k1 if np.isfinite(k)]; k2=[k for k in k2 if np.isfinite(k)]
        cells.append(f'{kind.split("-")[0][:6]:6s} {np.median(k1) if k1 else float("nan"):+.2f}->{np.median(k2) if k2 else float("nan"):+.2f}')
    print(f'  x {e[:12]:13s} ' + '  '.join(cells))
print('=== hermes-4 yes-rates: v1 -> LP on the v1-floor metrics + overall ===')
for n in sorted(H_v1[0]):
    if any(n.startswith(s) for s in ('Brand identity','PLANTED-length-long','PLANTED-digit')):
        print(f'  {n[:50]:52s} {np.nanmean(H_v1[0][n]):.2f} -> {np.nanmean(H_lp[0][n]):.2f}')
y1=[np.nanmean(v) for v in H_v1[0].values()]; y2=[np.nanmean(v) for v in H_lp[0].values()]
print(f'  name-arm yes median: v1 {np.median(y1):.2f} -> LP {np.median(y2):.2f}')
n1=np.mean([np.mean(~np.isfinite(v)) for v in H_v1[0].values()])
print(f'  nan rate v1 {n1:.3f}')
print('=== llama1b-or fam progress ===')
import subprocess
print(subprocess.run(['tail','-1',f'{O}/logs/or_fam_llama1b.log'],capture_output=True,text=True).stdout.strip())

set -u
cd "$(dirname "$0")"
until ssh sk3 'grep -q DENSE_STANDARD_SCALEUPC_DONE /lfs/skampere3/0/alexspan/norm-research/logs/mathse_vote_seeds12_gpu4.log' 2>/dev/null; do sleep 120; done
echo "=== SEEDS DONE $(date) ==="
ssh sk3 'cat /lfs/skampere3/0/alexspan/norm-research/datasets/math-stackexchange/v2_va/dense_standard_mathse_vote_score/eval_pass_results.json'
python3 fetch_dense.py 2>&1 | tail -2
python3 - <<'PY'
import json, numpy as np, warnings; warnings.filterwarnings('ignore')
import cells as C
from sklearn.metrics import roc_auc_score
d = C.load(); VA = d['layer1']['ledger']['VA_nl_mean']
ev = d['dense_split'] == 'eval'; te = d['dense_split'] == 'test'; held = ev | te
y = d['y']
def per(mask):
    return [float(roc_auc_score(y[mask], d['dense_seeds'][mask, j]))
            for j in range(d['dense_seeds'].shape[1])]
pe, pt, ph = per(ev), per(te), per(held)
g = {'seeds': d['dense_seed_ids'], 'VA_nl_mean3': VA,
     'T_eval_per_seed': pe, 'T_eval_mean': float(np.mean(pe)),
     'T_test_per_seed': pt, 'T_test_mean': float(np.mean(pt)),
     'T_eval_plus_test_per_seed': ph, 'T_eval_plus_test_mean': float(np.mean(ph))}
g['GATE_Delta_ledger_convention_eval'] = g['T_eval_mean'] - VA
g['Delta_test_only'] = g['T_test_mean'] - VA
g['Delta_eval_plus_test'] = g['T_eval_plus_test_mean'] - VA
g['GATE_PASS_gt_02'] = bool(g['GATE_Delta_ledger_convention_eval'] > 0.02)
g['rule'] = "proceed iff mean over seeds {42,1,2} of EVAL AUC minus VA_nl_mean3 > .02"
json.dump(g, open('gate_3seed.json', 'w'), indent=1)
print(json.dumps(g, indent=1))
PY
echo "=== REFRESHING ROUND 0 AT 3 SEEDS ==="
OMP_NUM_THREADS=6 bash refresh_round0_3seed.sh 2>&1 | tail -40

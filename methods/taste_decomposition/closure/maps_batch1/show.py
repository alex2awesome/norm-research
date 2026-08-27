import json, sys
c, r = sys.argv[1], sys.argv[2]
d = json.load(open(f"{c}_r{r}_results.json"))
print(f"=== {c} r{r} ===")
ta = d['track_A']
print(f"  gain MON {ta['gain_MONITOR']:+.4f}  gain HON {ta['gain_HONEST']:+.4f}  "
      f"D_beyond HON {ta['Delta_beyond_HONEST_new']:+.4f} (prev {ta['Delta_beyond_HONEST_prev']:+.4f})")
print(f"  spurious-alone joint HONEST {d['discount_ALL_B']['spurious_alone_AUC_histgb_HONEST']:.3f}")
for ch in d['spurious_map']['channels']:
    print(f"    {ch['alone_AUC_HONEST']:.3f} {'MIX' if ch['mixed'] else '   '} "
          f"{ch['name'][:52]:52s} <- {ch['upstream_parent'][:34]}")
s = d['stacked_increment_HONEST']
print(f"  STACK HON: B {s['AUC_jointB']:.3f} dense {s['AUC_dense']:.3f} bank {s['AUC_bank_VA_nl']:.3f}"
      f" | dense incr {s['dense_increment_over_B']:+.4f} p={s['ci_dense_increment']['p_gt0']:.2f}"
      f" | bank incr {s['bank_increment_over_B']:+.4f} p={s['ci_bank_increment']['p_gt0']:.2f}")
for k in ('discount_ALL_B', 'discount_STRICT_no_mixed'):
    x = d.get(k)
    if x:
        j = x['stratified_HONEST_q10']['joint_B_score']
        print(f"  {k:26s} T_adj {j['T_adj']:.3f} VA_adj {j['VA_adj']:.3f} "
              f"D_adj {j['Delta_adj']:+.4f} (strata {j['n_strata_used']}, rows {j['n_rows_used']})")
sr = d.get('score_report') or {}
a = sr.get('anchors', {})
print(f"  anchors pos-vs-neg {a.get('pos_vs_neg_auc',0):.3f} coh-vs-scram {a.get('coherent_vs_scrambled_auc',0):.3f} "
      f"collapsed {sr.get('n_collapsed')} NA {sr.get('overall_na_rate',0):.3f}")

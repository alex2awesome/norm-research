#!/usr/bin/env python3
"""ANALYSIS 1 -- "swap" decomposition of the Layer-3 closure pilot.

QUESTION.  Across rounds 1-3 the articulated bank's rank agreement with the dense
model rose (Spearman rho .488 -> .541 on MONITOR) while its label AUC stayed
essentially flat (+.010 over four rounds).  Did the bank SWAP correct examples --
gaining ordered pairs the dense model gets right while LOSING pairs the bank had
right that the dense model gets wrong?

METHOD.  AUC on a binary-label population is exactly the fraction of concordant
(positive, negative) pairs.  So partition the pairs by DENSE correctness and read
the bank inside each cell:

    D+  : dense ranks the accepted paper above the rejected one   (weight w+)
    D-  : dense has it backwards                                  (weight w-)
    D0  : dense ties them                                         (weight w0)

    C+  = P(bank concordant with truth | D+)
    C-  = P(bank concordant with truth | D-)

    AUC_bank      = w+ C+ + w0 C0 + w- C-
    AUC_dense     = w+ + .5 w0
    agreement     = w+ C+ + .5 w0 + w- (1 - C-)      <- bank/dense order agreement
                                                        on label-DISCORDANT pairs

Agreement with dense on D+ is exactly C+ (agreeing with dense = being right).
Agreement with dense on D- is exactly 1 - C- (agreeing with dense = being wrong).
So the two "inheritance" rates the swap hypothesis is about are directly readable,
and AUC and agreement are two different linear functionals of the same (C+, C-).

  * flat AUC + rising agreement REQUIRES  w+ dC+ ~= -w- dC-,  i.e. the bank must
    take on dense's errors at (w+/w-) ~ 3.5x the per-pair rate at which it takes
    on dense's insights.  That is the SWAP signature.
  * the alternative is that the rho rise lives on SAME-LABEL (pos,pos)/(neg,neg)
    pairs, which contribute to rho but are invisible to AUC -- pure non-diagnostic
    redundancy.  Both are measured here.

POPULATION.  The honest one: the 1,244 dense-held-out rows (dense_split in
{eval,test}), with same-rows dense predictions from peer_verdict_dense_preds.csv.
Bank predictions are out-of-sample everywhere: grouped-OOF inside FIT+MINE,
refit-and-predict on MONITOR -- exactly the `va_full` construction the pilot's
`Delta_honest_level_heldout1244` uses.  MONITOR-only figures are reported
alongside for continuity with the published rho series and are flagged as
dense-CONTAMINATED (943/1,192 MONITOR rows were in the dense model's train split).

Uncertainty: group-level (ntitle) paired bootstrap, 2,000 draws, matching the
pilot's `group_boot_ci`.

CPU only.  Requires round_preds_all.npz (see recompute_round_preds.py).
Usage: python swap_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROUNDS = ["round0", "round1", "round2", "round3", "round4"]
N_BOOT = 2000
SEED = 0


# --------------------------------------------------------------- pair algebra --
def build_pair_arrays(va, dense, y):
    """Flattened (pos, neg) pair design plus the same-label pair design."""
    P = np.where(y == 1)[0]
    N = np.where(y == 0)[0]
    D = np.sign(dense[P][:, None] - dense[N][None, :]).astype(np.float32)
    masks = {"Dp": (D > 0), "Dm": (D < 0), "D0": (D == 0)}

    layers, names = [], []
    for mk, mv in masks.items():
        layers.append(mv.astype(np.float32))
        names.append(mk)
    conc = {}
    for r in range(len(ROUNDS)):
        S = np.sign(va[r][P][:, None] - va[r][N][None, :]).astype(np.float32)
        c = (S > 0).astype(np.float32) + 0.5 * (S == 0)
        conc[r] = c
        for mk, mv in masks.items():
            layers.append((mv * c).astype(np.float32))
            names.append(f"c{r}_{mk}")
    M = np.stack(layers).reshape(len(layers), -1)  # (K, n_pos*n_neg)
    return P, N, M, names, conc, masks


def build_samelabel_arrays(va, dense, y):
    """Upper-triangle (i<j) pairs within each label class."""
    out_i, out_j = [], []
    for lab in (1, 0):
        idx = np.where(y == lab)[0]
        ii, jj = np.triu_indices(len(idx), k=1)
        out_i.append(idx[ii])
        out_j.append(idx[jj])
    I = np.concatenate(out_i)
    J = np.concatenate(out_j)
    dd = np.sign(dense[I] - dense[J])
    layers = [np.ones(len(I), dtype=np.float32)]
    names = ["all"]
    for r in range(len(ROUNDS)):
        ss = np.sign(va[r][I] - va[r][J])
        agree = (ss == dd).astype(np.float32) + 0.5 * ((ss == 0) | (dd == 0)) * (ss != dd)
        layers.append(agree)
        names.append(f"agree{r}")
    return I, J, np.stack(layers), names


def pair_stats(vals, names):
    """Turn the weighted layer sums into the reported quantities."""
    g = dict(zip(names, vals))
    tot = g["Dp"] + g["Dm"] + g["D0"]
    out = {"w_plus": g["Dp"] / tot, "w_minus": g["Dm"] / tot, "w_tie": g["D0"] / tot}
    out["AUC_dense"] = out["w_plus"] + 0.5 * out["w_tie"]
    for r in range(len(ROUNDS)):
        Cp = g[f"c{r}_Dp"] / g["Dp"] if g["Dp"] > 0 else np.nan
        Cm = g[f"c{r}_Dm"] / g["Dm"] if g["Dm"] > 0 else np.nan
        C0 = g[f"c{r}_D0"] / g["D0"] if g["D0"] > 0 else 0.5
        out[f"C_plus_r{r}"] = Cp
        out[f"C_minus_r{r}"] = Cm
        out[f"agree_dense_on_Dplus_r{r}"] = Cp          # = C+
        out[f"agree_dense_on_Dminus_r{r}"] = 1.0 - Cm   # = 1 - C-
        out[f"AUC_bank_r{r}"] = (g[f"c{r}_Dp"] + g[f"c{r}_Dm"] + g[f"c{r}_D0"]) / tot
        out[f"agree_discordant_r{r}"] = (
            g[f"c{r}_Dp"] + (g["Dm"] - g[f"c{r}_Dm"]) + 0.5 * g["D0"]
        ) / tot
    return out


STEPS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]


def contrasts(st, sl):
    """Round-to-round changes, including the identity that ties AUC to agreement.

    For any step a->b:
        d_AUC   = w+ dC+ + w- dC-        (dense-wrong pairs SUBTRACT if dC- < 0)
        d_agree = w+ dC+ - w- dC-        (dense-wrong pairs ADD if dC- < 0)
    so a step with dC+ > 0 and dC- < 0 of matched magnitude is a pure SWAP:
    agreement climbs, AUC does not.
    """
    wp, wm = st["w_plus"], st["w_minus"]
    c = {}
    for a, b in STEPS:
        t = f"r{a}_to_r{b}"
        dCp = st[f"C_plus_r{b}"] - st[f"C_plus_r{a}"]
        dCm = st[f"C_minus_r{b}"] - st[f"C_minus_r{a}"]
        c[f"{t}__dC_plus"] = dCp
        c[f"{t}__dC_minus"] = dCm
        c[f"{t}__d_anticonc_minus"] = -dCm      # = D(agreement with dense on D-)
        c[f"{t}__d_AUC"] = st[f"AUC_bank_r{b}"] - st[f"AUC_bank_r{a}"]
        c[f"{t}__d_agree_discordant"] = (st[f"agree_discordant_r{b}"]
                                         - st[f"agree_discordant_r{a}"])
        c[f"{t}__contrib_AUC_from_Dplus"] = wp * dCp
        c[f"{t}__contrib_AUC_from_Dminus"] = wm * dCm
        c[f"{t}__contrib_agree_from_Dplus"] = wp * dCp
        c[f"{t}__contrib_agree_from_Dminus"] = wm * (-dCm)
        # >0 = the step took on dense's ERRORS faster than dense's insights
        c[f"{t}__error_minus_insight_inheritance"] = (-dCm) - dCp
        c[f"{t}__d_agree_samelabel"] = sl[b] - sl[a]
    for r in range(len(ROUNDS)):
        c[f"agree_samelabel_r{r}"] = sl[r]
    return c


# ------------------------------------------------------------------- bootstrap -
def group_weights(rng, groups_codes, n_groups, n_rows):
    draw = rng.integers(0, n_groups, size=n_groups)
    mult = np.bincount(draw, minlength=n_groups).astype(np.float32)
    return mult[groups_codes]


def main():
    z = np.load(HERE / "round_preds_all.npz", allow_pickle=True)
    y_all = z["y"]
    nt_all = z["ntitle"]
    dense_all = z["dense"]
    held = z["held"].astype(bool)
    monm = z["monitor"].astype(bool)

    res = {
        "analysis": "swap decomposition of the Layer-3 closure pilot (peer-review verdict)",
        "population": {
            "primary": "honest = 1,244 dense-held-out rows (dense_split in {eval,test}); "
                       "bank preds out-of-sample everywhere (grouped-OOF in FIT+MINE, "
                       "refit-predict on MONITOR)",
            "secondary_flagged": "MONITOR (n=1,192) -- dense preds CONTAMINATED "
                                 "(943/1,192 rows in the dense model's train split); "
                                 "reported only to reproduce the published rho series",
        },
        "n_boot": N_BOOT,
    }

    for tag, mask, contaminated in (("honest_heldout1244", held, False),
                                    ("MONITOR_all", monm, True)):
        y = y_all[mask]
        nt = nt_all[mask]
        dense = dense_all[mask]
        va = np.array([z[f"va_nl_{r}"][mask] for r in ROUNDS])

        # ---- Spearman rho series (the statistic the question is about) --------
        rho = [float(spearmanr(va[r], dense).statistic) for r in range(len(ROUNDS))]
        auc_pt = None

        P, N, M, names, conc, masks = build_pair_arrays(va, dense, y)
        I, J, SL, sl_names = build_samelabel_arrays(va, dense, y)

        wp1 = np.ones(len(P), dtype=np.float32)
        wn1 = np.ones(len(N), dtype=np.float32)
        w_out = np.outer(wp1, wn1).ravel()
        st = pair_stats(M @ w_out, names)
        w_sl = np.ones(len(I), dtype=np.float32)
        sl_sum = SL @ w_sl
        sl_pt = [float(sl_sum[1 + r] / sl_sum[0]) for r in range(len(ROUNDS))]
        ct = contrasts(st, sl_pt)

        # ---- flow tables, cross-tabbed by dense correctness -------------------
        flows = {}
        for a, b in STEPS:
            righta = conc[a] > 0.5
            rightb = conc[b] > 0.5
            flow = {}
            for mk in ("Dp", "Dm", "D0"):
                m = masks[mk]
                n = int(m.sum())
                cell = {
                    "n_pairs": n,
                    "right_to_right": int((m & righta & rightb).sum()),
                    "right_to_wrong": int((m & righta & ~rightb).sum()),
                    "wrong_to_right": int((m & ~righta & rightb).sum()),
                    "wrong_to_wrong": int((m & ~righta & ~rightb).sum()),
                }
                bw = cell["wrong_to_right"] + cell["wrong_to_wrong"]
                br = cell["right_to_right"] + cell["right_to_wrong"]
                cell["gain_rate_among_start_wrong"] = cell["wrong_to_right"] / bw if bw else np.nan
                cell["loss_rate_among_start_right"] = cell["right_to_wrong"] / br if br else np.nan
                cell["net_flips_per_pair"] = (cell["wrong_to_right"] - cell["right_to_wrong"]) / n if n else np.nan
                flow[mk] = cell
            tg = sum(flow[k]["wrong_to_right"] for k in ("Dp", "Dm", "D0"))
            tl = sum(flow[k]["right_to_wrong"] for k in ("Dp", "Dm", "D0"))
            npairs = int(masks["Dp"].size)
            flow["totals"] = {
                "gained_pairs": tg, "lost_pairs": tl, "net": tg - tl,
                "share_of_gains_in_Dplus": flow["Dp"]["wrong_to_right"] / tg if tg else np.nan,
                "share_of_losses_in_Dminus": flow["Dm"]["right_to_wrong"] / tl if tl else np.nan,
                "share_of_pairs_that_are_Dplus": float(masks["Dp"].mean()),
                "share_of_pairs_that_are_Dminus": float(masks["Dm"].mean()),
                # >1 = losses over-represented among the pairs dense gets wrong
                "loss_enrichment_in_Dminus": (
                    (flow["Dm"]["right_to_wrong"] / tl) / float(masks["Dm"].mean())
                    if tl and masks["Dm"].mean() else np.nan),
                "gain_enrichment_in_Dplus": (
                    (flow["Dp"]["wrong_to_right"] / tg) / float(masks["Dp"].mean())
                    if tg and masks["Dp"].mean() else np.nan),
                "churn_rate": (tg + tl) / npairs,
            }
            flows[f"r{a}_to_r{b}"] = flow
        flow = flows["r0_to_r4"]

        # ---- bootstrap -------------------------------------------------------
        rng = np.random.default_rng(SEED)
        uniq, codes = np.unique(nt, return_inverse=True)
        ng = len(uniq)
        boot_st, boot_ct = [], []
        for _ in range(N_BOOT):
            w = group_weights(rng, codes, ng, len(y))
            wp, wn = w[P], w[N]
            if wp.sum() == 0 or wn.sum() == 0:
                continue
            s = pair_stats(M @ np.outer(wp, wn).ravel(), names)
            slw = w[I] * w[J]
            ss = SL @ slw
            slb = [float(ss[1 + r] / ss[0]) if ss[0] > 0 else np.nan for r in range(len(ROUNDS))]
            boot_st.append(s)
            boot_ct.append(contrasts(s, slb))

        def ci(dicts, key):
            a = np.array([d[key] for d in dicts], dtype=float)
            a = a[np.isfinite(a)]
            return {"lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)),
                    "p_gt0": float((a > 0).mean()), "boot_mean": float(a.mean())}

        block = {
            "n_rows": int(mask.sum()),
            "n_pos": int(len(P)), "n_neg": int(len(N)),
            "n_discordant_pairs": int(len(P) * len(N)),
            "n_samelabel_pairs": int(len(I)),
            "dense_contaminated": contaminated,
            "spearman_rho_VAnl_vs_dense": {ROUNDS[r]: rho[r] for r in range(len(ROUNDS))},
            "point": {k: float(v) for k, v in st.items()},
            "samelabel_agreement": {ROUNDS[r]: sl_pt[r] for r in range(len(ROUNDS))},
            "contrasts": {k: float(v) for k, v in ct.items()},
            "contrast_CIs": {k: ci(boot_ct, k) for k in ct
                             if not k.startswith("agree_samelabel_r")},
            "per_round_CIs": {k: ci(boot_st, k) for k in
                              [f"C_plus_r{r}" for r in range(5)] +
                              [f"C_minus_r{r}" for r in range(5)] +
                              [f"AUC_bank_r{r}" for r in range(5)] +
                              [f"agree_discordant_r{r}" for r in range(5)]},
            "flow": flows,
        }
        res[tag] = block
        print(f"=== {tag} (n={block['n_rows']}) ===")
        print("  rho:", [round(x, 4) for x in rho])
        print("  AUC_dense:", round(st["AUC_dense"], 4),
              " w+/w-/w0:", round(st["w_plus"], 4), round(st["w_minus"], 4), round(st["w_tie"], 4))
        print("  C+ :", [round(st[f"C_plus_r{r}"], 4) for r in range(5)])
        print("  C- :", [round(st[f"C_minus_r{r}"], 4) for r in range(5)])
        print("  AUC:", [round(st[f"AUC_bank_r{r}"], 4) for r in range(5)])
        print("  agr(disc):", [round(st[f"agree_discordant_r{r}"], 4) for r in range(5)])
        print("  agr(same-label):", [round(x, 4) for x in sl_pt])
        print("  --- per-step decomposition (dC+ | dC- | dAUC | dAgr | err-insight) ---")
        for a, b in STEPS:
            t = f"r{a}_to_r{b}"
            ciA = block["contrast_CIs"][f"{t}__d_AUC"]
            ciE = block["contrast_CIs"][f"{t}__error_minus_insight_inheritance"]
            print(f"  {t}: {ct[f'{t}__dC_plus']:+.4f} | {ct[f'{t}__dC_minus']:+.4f} | "
                  f"{ct[f'{t}__d_AUC']:+.4f} [{ciA['lo']:+.4f},{ciA['hi']:+.4f}] | "
                  f"{ct[f'{t}__d_agree_discordant']:+.4f} | "
                  f"{ct[f'{t}__error_minus_insight_inheritance']:+.4f} "
                  f"[{ciE['lo']:+.4f},{ciE['hi']:+.4f}] P(>0)={ciE['p_gt0']:.2f}")
            ft = flows[t]["totals"]
            print(f"      flow: gained {ft['gained_pairs']} lost {ft['lost_pairs']} "
                  f"net {ft['net']} | loss-enrichment in D- = {ft['loss_enrichment_in_Dminus']:.2f}x "
                  f"| gain-enrichment in D+ = {ft['gain_enrichment_in_Dplus']:.2f}x")

    (HERE / "swap_analysis.json").write_text(json.dumps(res, indent=2))
    print("wrote", HERE / "swap_analysis.json")


if __name__ == "__main__":
    main()

### A. Stacked increment on the HONEST population — the batch's central table

| cell | round | AUC(named nuisance model) | AUC(dense) | AUC(bank) | dense increment over nuisance | p | bank increment over nuisance | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| peer_curation | r1 | 0.545 | 0.594 | 0.570 | **+0.0452** | 0.99 | **+0.0141** | 0.80 |
| peer_curation | r2 | 0.561 | 0.594 | 0.577 | **+0.0316** | 0.98 | **+0.0118** | 0.81 |
| peer_revealed | r1 | 0.752 | 0.884 | 0.751 | **+0.1331** | 1.00 | **+0.0200** | 0.97 |
| peer_revealed | r2 | 0.768 | 0.884 | 0.763 | **+0.1188** | 1.00 | **+0.0318** | 1.00 |
| cap_crowd | r1 | 0.545 | 0.555 | 0.630 | **+0.0105** | 0.83 | **+0.0845** | 1.00 |
| cap_crowd | r2 | 0.580 | 0.555 | 0.634 | **+0.0052** | 0.79 | **+0.0564** | 1.00 |
| cap_finalist | r1 | 0.646 | 0.612 | 0.679 | **+0.0038** | 0.71 | **+0.0303** | 0.99 |
| cap_finalist | r2 | 0.598 | 0.612 | 0.664 | **+0.0273** | 0.97 | **+0.0650** | 1.00 |
| nc_outcome | r1 | 0.637 | 0.624 | 0.632 | **+0.0152** | 0.95 | **+0.0071** | 0.83 |
| nc_outcome | r2 | 0.621 | 0.624 | 0.625 | **+0.0263** | 1.00 | **+0.0248** | 0.99 |
| nc_agree | r1 | 0.596 | 0.603 | 0.624 | **+0.0193** | 0.91 | **+0.0248** | 0.93 |
| nc_agree | r2 | 0.607 | 0.603 | 0.632 | **+0.0203** | 0.94 | **+0.0269** | 0.98 |

### B. Discount: Δ_adj with and without the MIXED channels (Addendum-2 sensitivity band)

| cell | round | pooled Δ (HONEST) | Δ_adj ALL B | Δ_adj STRICT | band width | T_adj ALL | VA_adj ALL |
|---|---|---:|---:|---:|---:|---:|---:|
| peer_curation | r1 | +0.0233 | +0.0288 | +0.0295 | 0.0007 | 0.591 | 0.562 |
| peer_curation | r2 | +0.0163 | +0.0212 | +0.0130 | 0.0082 | 0.583 | 0.562 |
| peer_revealed | r1 | +0.1331 | +0.1974 | +0.1394 | 0.0580 | 0.836 | 0.638 |
| peer_revealed | r2 | +0.1210 | +0.1686 | +0.1336 | 0.0350 | 0.842 | 0.673 |
| cap_crowd | r1 | -0.0748 | -0.0853 | -0.0744 | 0.0109 | 0.541 | 0.626 |
| cap_crowd | r2 | -0.0783 | -0.0786 | -0.0765 | 0.0021 | 0.543 | 0.621 |
| cap_finalist | r1 | -0.0668 | -0.1139 | -0.0815 | 0.0324 | 0.501 | 0.615 |
| cap_finalist | r2 | -0.0518 | -0.0573 | -0.0558 | 0.0015 | 0.582 | 0.640 |
| nc_outcome | r1 | -0.0084 | +0.0169 | -0.0011 | 0.0179 | 0.578 | 0.561 |
| nc_outcome | r2 | -0.0017 | +0.0044 | +0.0045 | 0.0001 | 0.583 | 0.578 |
| nc_agree | r1 | -0.0205 | -0.0217 | -0.0168 | 0.0049 | 0.576 | 0.597 |
| nc_agree | r2 | -0.0284 | -0.0231 | -0.0340 | 0.0109 | 0.579 | 0.602 |

### C. Track A closure (secondary): per-round VA_nl gain

| cell | r1 gain HONEST | r1 gain MONITOR | r2 gain HONEST | r2 gain MONITOR | cleared ε=.005? | Δ_beyond HONEST after r2 |
|---|---:|---:|---:|---:|---|---:|
| peer_curation | -0.0119 | -0.0111 | +0.0071 | +0.0048 | r1 no / r2 yes | +0.0163 |
| peer_revealed | +0.0271 | +0.0066 | +0.0121 | +0.0142 | r1 yes / r2 yes | +0.1210 |
| cap_crowd | +0.0067 | +0.0063 | +0.0035 | -0.0023 | r1 yes / r2 no | -0.0783 |
| cap_finalist | +0.0097 | +0.0155 | -0.0151 | +0.0085 | r1 yes / r2 no | -0.0518 |
| nc_outcome | -0.0001 | +0.0048 | -0.0067 | -0.0113 | r1 no / r2 no | -0.0017 |
| nc_agree | -0.0019 | -0.0110 | +0.0079 | -0.0003 | r1 no / r2 yes | -0.0284 |

### D. Missing mass, both tracks (FREEZE ADDENDUM: the B-side species machinery)

| cell | round | A: S_obs | A: M̂ | A: LOPO | A: recapture | B: S_obs | B: M̂ | B: LOPO | B: recapture |
|---|---|---:|---:|---|---:|---:|---:|---|---:|
| peer_curation | r1 | 54 | 0.817 | [0.76, 0.96] | 0.09 | 28 | **0.550** | [0.57, 0.67] | 0.21 |
| peer_curation | r2 | 46 | 0.633 | [0.53, 0.91] | 0.17 | 30 | **0.650** | [0.60, 0.73] | 0.13 |
| peer_revealed | r1 | 50 | 0.750 | [0.76, 0.82] | 0.10 | 27 | **0.500** | [0.50, 0.70] | 0.26 |
| peer_revealed | r2 | 50 | 0.700 | [0.71, 0.78] | 0.12 | 31 | **0.675** | [0.63, 0.73] | 0.13 |
| cap_crowd | r1 | 42 | 0.417 | [0.22, 0.78] | 0.33 | 34 | **0.750** | [0.73, 0.87] | 0.12 |
| cap_crowd | r2 | 39 | 0.500 | [0.44, 0.67] | 0.23 | 27 | **0.500** | [0.47, 0.60] | 0.22 |
| cap_finalist | r1 | 34 | 0.400 | [0.38, 0.53] | 0.26 | 30 | **0.600** | [0.60, 0.70] | 0.20 |
| cap_finalist | r2 | 35 | 0.383 | [0.44, 0.49] | 0.31 | 29 | **0.575** | [0.53, 0.67] | 0.17 |
| nc_outcome | r1 | 29 | 0.383 | [0.36, 0.51] | 0.21 | 35 | **0.775** | [0.77, 0.87] | 0.11 |
| nc_outcome | r2 | 36 | 0.500 | [0.38, 0.69] | 0.17 | 33 | **0.650** | [0.73, 0.80] | 0.21 |
| nc_agree | r1 | 29 | 0.350 | [0.22, 0.56] | 0.28 | 27 | **0.450** | [0.43, 0.57] | 0.30 |
| nc_agree | r2 | 30 | 0.450 | [0.31, 0.58] | 0.10 | 34 | **0.750** | [0.70, 0.87] | 0.12 |

### E. Scoring-batch quality gates (every round)

| cell | round | anchors coherent-vs-scrambled (gate ≥.70) | anchors pos-vs-neg | collapsed criteria | NA rate | routing misroute | probes |
|---|---|---:|---:|---:|---:|---:|---:|
| peer_curation | r1 | 0.933 PASS | 0.589 | 0 | 0.006 | 0.04 | 4/4 |
| peer_curation | r2 | 0.979 PASS | 0.617 | 0 | 0.024 | 0.08 | 4/4 |
| peer_revealed | r1 | 0.970 PASS | 0.642 | 0 | 0.013 | 0.04 | 4/4 |
| peer_revealed | r2 | 0.972 PASS | 0.641 | 0 | 0.011 | 0.12 | 4/4 |
| cap_crowd | r1 | 1.000 PASS | 0.622 | 2 | 0.006 | 0.00 | 2/4 |
| cap_crowd | r2 | 0.981 PASS | 0.584 | 0 | 0.003 | 0.24 | 4/4 |
| cap_finalist | r1 | 1.000 PASS | 0.398 | 0 | 0.002 | 0.04 | 4/4 |
| cap_finalist | r2 | 0.984 PASS | 0.368 | 1 | 0.002 | 0.00 | 4/4 |
| nc_outcome | r1 | 0.945 PASS | 0.718 | 0 | 0.002 | 0.04 | 4/4 |
| nc_outcome | r2 | 0.995 PASS | 0.667 | 0 | 0.002 | 0.04 | 4/4 |
| nc_agree | r1 | 0.983 PASS | 0.590 | 0 | 0.005 | 0.12 | 4/4 |
| nc_agree | r2 | 0.873 PASS | 0.606 | 0 | 0.002 | 0.16 | 4/4 |

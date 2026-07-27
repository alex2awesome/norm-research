# Vector scale–articulation substitution: the fixed-policy M_omega join

Date: 2026-07-12  
Status: formal synthesis after the first direct-policy lockbox. Additive to the existing name-
sufficiency, unit-count, decompression, CUF, and prompt-optimality lines; none is overwritten.

## 1. The elegant join

Let a larger reader `B` receiving a sparse address `n` define the fixed target channel

```
Q(x, phi) = P_B(YES | x, n, phi).
```

Let a smaller executor `S` receive an explicit articulation `h`:

```
P_h(x, phi) = P_S(YES | x, h, phi).
```

Then "can tacit knowledge made explicit replace scale?" is exactly a fixed-target prompt-
reconstruction problem. `Q` takes the role of the fixed `M_omega` channel. It is not an external
ground truth: it is the larger model's own name-only policy. Definitions, explanations, rules,
interactions, boundary cases, and ostensive contrasts are candidate encodings of that policy for the
smaller executor.

This imports the strongest parts of prompt-optimality unchanged:

- a fixed target channel and a held-out reconstruction objective;
- form quotients and adverse-form audits;
- the fresh-development/selection/lockbox discipline;
- behavioral rather than lexical identity;
- CUF fingerprints, executor-relative units, redundancy/interaction analysis, and non-monotone
  composition;
- the fixed-target DPI fact `M_Q <- X -> Z_(E,h)`, hence
  `I_f(M_Q;Z_(E,h)) <= I_f(M_Q;X)`. This is a cap from target information available through the
  item channel `X`, not from the fixed message `h` itself: a global articulation has no
  item-varying information until an executor applies it to `X`.

It does **not** import the old `OPT_Omega + epsilon` checklist ceiling as a ceiling on this target.
That bridge was retracted: enumerating known prompt units does not upper-bound the impact of missing
or gestalt content. The direct identity test below supplies the relevant endpoint instead.

## 2. Isomorphism is a region, not one scalar

Average the target's meaning-preserving form orbit:

```
q_i = mean_phi Q(x_i, phi).
```

For each candidate, define the adverse coordinate vector

```
L(h;Q) = (
  max_phi mean_i |P_h(x_i,phi) - q_i|,              # Bernoulli TVD / MAE
  1 - min_phi Spearman(P_h(x_i,phi), q_i),          # item-order loss
  max_phi Flip_0.5(P_h(x_i,phi), q_i),              # threshold loss
  max_phi |mean_i(P_h(x_i,phi)-q_i)|                # level bias
).
```

The target has its own form-identity radius `R(Q)` in the same coordinates. The isomorphism set is

```
I_Q(epsilon) = { h : paired uncertainty places L(h;Q)-R(Q) inside epsilon
                       and every form has positive target polarity }.
```

Exact scale substitution means `h in I_Q(epsilon)` while the smaller name-only policy is outside it.
An equal-but-different articulation fiber is the behaviorally and semantically audited preimage of
this set. Diversity is evaluated only after membership; equal aggregate scores do not create a
fiber.

That target-self region is the **near-identity** tier. A nested functional ordinal tier uses the
absolute rank-loss tolerance requested for approximate isomorphism:

```
F_Q(0.30) = {h : min_phi Spearman(P_h^phi, q) >= 0.70,
                  Spearman(mean_phi P_h^phi, q) >= 0.70,
                  positive target polarity }.
```

An observed member clears the `.70` point floor; a certified member's bootstrap lower bound clears
both the adverse-form and quotient floors. The latter is the appropriate confirmatory grade.
Functional **substitution** further requires direct MAE improvement and a smaller sparse/name
baseline below the same floor at the corresponding point/interval grade. This tier was introduced
after the first G6 lockbox was opened, so all current re-reads are retrospective and cannot
overwrite its frozen near-identity conclusion.

When a fixed top target `Q` and an intermediate larger executor `E` coexist, there is a second
policy object that must not be collapsed into target-relative performance. Let

```
r_E(x) = mean_phi P_E(YES | x, n, phi)
```

be the larger sparse endpoint itself. The accepted functional endpoint region is

```
F_E(0.30) = {h : min_phi Spearman(P_(e,h)^phi, r_E) >= 0.70,
                  Spearman(mean_phi P_(e,h)^phi, r_E) >= 0.70,
                  positive endpoint polarity}.
```

This tests direct behavioral fidelity to the larger policy. A **direct functional endpoint
substitution** additionally requires that the articulation improve direct MAE over the smaller
sparse policy and that the smaller sparse policy's predeclared adverse-rank coordinate lie below
`.70`. The quotient-rank and any-coordinate baseline diagnostics are reported but do not enter that
exclusion gate; this avoids an undeclared union test. The observed grade uses points and the
certified grade uses the corresponding interval edges.

Direct endpoint fidelity is different from one-sided noninferiority against `Q`, which can be
passed by overshooting `E` while producing a different item policy. A separate two-sided
equal-target-loss region is

```
E_Q(delta) = {h : |L_j(e,h;Q) - L_j(E,n;Q)| <= delta_j
                    for j in {adverse rank, adverse MAE}}.
```

The implemented primary equivalence grade uses the robust/adverse rank and MAE loss envelope; a
vector sensitivity grade additionally requires flips and bias. It is not matched-form equality,
and equal loss to `Q` does not imply equal item policy. In particular, neither one-sided
noninferiority nor two-sided target-loss equivalence is a weaker version of direct endpoint
isomorphism: they are different relations and can disagree.

The local evidence is therefore a partially ordered set, not one nested ladder:

- one-sided target-relative noninferiority recovery;
- two-sided target-relative equal-loss recovery `E_Q`;
- direct functional endpoint fidelity `F_E`, upgraded to substitution only by direct MAE gain and
  the predeclared adverse-rank baseline exclusion;
- direct endpoint near-identity in the larger endpoint's self band, nested under `F_E` and upgraded
  to substitution by its corresponding baseline exclusion.

Every scale-step grade also requires a genuine native rank/MAE gap and articulation rank/MAE gain.
The strongest canonical-v4 functional join is the conjunction of fixed-target substitution in
`F_Q` and direct-endpoint substitution in `F_E`; its stricter version also requires `E_Q`. Matched
inert and wrong-construct controls are a separate content-specificity requirement, not silently
part of that v4 code-level conjunction. Replacing the functional regions with target-self bands is
the near-cloning claim. This structure lets `.70` remain the declared functional epsilon without
relabeling one-sided noninferiority or equal target loss as policy isomorphism.

This geometry explains why a universal scalar law was too coarse. Articulation can move different
coordinates in different directions. More content can fix policy level while scrambling order, or
improve rank while shifting the threshold. A single AUC or a single unit count hides that structure.

## 3. Two substitution vectors

There are two useful, explicitly different normalizations.

### 3.1 Direct target closure

For lower-is-better coordinate `j`, with the quotient target itself at zero loss,

```
lambda_raw,j(h) = [L_j(S,name;Q) - L_j(S,h;Q)] / L_j(S,name;Q).
```

For Spearman,

```
lambda_raw,rho(h) = [rho(S,h;Q)-rho(S,name;Q)] / [1-rho(S,name;Q)].
```

`1` is complete point recovery, `0` is no movement, a negative value is anti-substitution, and a
value above `1` is overshoot.

### 3.2 Excess beyond the target identity band

For lower-is-better coordinate `j`,

```
lambda_band,j(h) = [L_j(S,name;Q)-L_j(S,h;Q)]
                   / [L_j(S,name;Q)-R_j(Q)].
```

This asks how much of the *non-identity excess* articulation removes, rather than how much raw
distance to a mathematically perfect quotient it removes. It is the right bridge to the exact
certificate, but it never replaces the joint gate.

The scalar scale–articulation law is now a special case: an articulation operator behaves like a
horizontal scale shift only if its vector is stably positive and commensurate across coordinates,
constructs, model pairs, and forms. Universal exact substitution further requires simultaneous
identity-band entry, not merely `lambda_j > 0` on an average coordinate.

### 3.3 Actual scale-step estimands

The two `lambda` quantities above measure debt to the target itself. They must not be mistaken for
the amount of one **executor step** replaced. For a fixed target `Q`, smaller executor `e`, larger
executor `E`, and sparse arm `n`, define for every lower-is-better coordinate `j`:

```
S_j(e,E) = L_j(e,n;Q) - L_j(E,n;Q)       # native scale advantage
A_j(e,h) = L_j(e,n;Q) - L_j(e,h;Q)       # articulation gain
Gamma_j  = A_j(E,h) - A_j(e,h)           # executor x articulation interaction
```

Then the scale-step claim has an exact algebraic criterion:

```
L_j(e,h;Q) <= L_j(E,n;Q) + delta_j
    iff
A_j(e,h) >= S_j(e,E) - delta_j.
```

Joint **one-sided capability recovery** requires this preregistered comparison on every required
coordinate, positive polarity, and matched content-specificity controls. It does not by itself
establish equal performance or direct policy replication. A `.70` target-reconstruction result is
also not by itself a scale-step result. When `S_j>0`, the descriptive step-closure ratio is

```
chi_j(e->E,h) = A_j(e,h) / S_j(e,E).
```

It is undefined when the alleged larger executor has no positive native advantage on coordinate
`j`; it is also distinct from `lambda_band`, whose denominator is target-self debt. Certified
articulation debt is correspondingly

```
b*_(e->E)(Q,delta) = inf { cost(h): L(e,h;Q) <= L(E,n;Q)+delta },
```

with `+infinity`/right censoring when the frozen bank never enters the region. One passing arm
establishes an upper bound on debt; minimality needs simultaneous or sequential rejection of every
lower budget.

The frozen v4 `policy_isomorphism.py::certify_scale_step_substitution` implementation uses one shared
item/source-group bootstrap for `Q`, `(e,n)`, `(e,h)`, and `(E,n)`. It keeps five outcomes separate:
one-sided target-relative noninferiority; two-sided equal target loss; direct `.70` fidelity to the
`(E,n)` item policy and the stricter direct-substitution grade; direct target-self-band
near-identity; and absolute fixed-`Q` fidelity. The surrounding runner applies Bonferroni intervals
over eligible non-control arms and reports both joint fixed-target/direct-endpoint grades: the
functional-isomorphism conjunction and its stricter two-sided-equal-loss conjunction.

The reusable curve-collapse hypothesis is therefore conditional and vector-valued:

```
u_(c,j)(e,h) = f_(c,j)(kappa_(e,j) + a_(c,h,j) + gamma_(e,h,j)).
```

Horizontal scale/articulation collapse predicts small `gamma` and stable `a` within a construct and
coordinate. A universal scalar law is the much stronger nested hypothesis that the same `kappa`
and `a` work across coordinates and constructs. The observed rank/MAE separation makes the vector
model with executor interaction the live hypothesis and the scalar version a falsifiable special
case.

## 4. What the first lockbox establishes

For the best frozen 3B articulation of the 8B name-only **Wordplay quality and clarity** policy, the
400-item family-wise direct-quotient transport signature is

| coordinate | fraction of raw 3B-name gap closed (98.33% CI) |
|---|---:|
| MAE/TVD | **+.366** `[+.292,+.434]` |
| rank | **-.377** `[-.679,-.138]` |
| binary flips | **+.319** `[+.124,+.485]` |
| absolute bias | **+.963** `[+.865,+.999]` |

Relative to the 8B target's own adverse-form identity radius, the same signature is

| coordinate | fraction of non-identity excess removed (98.33% CI) |
|---|---:|
| MAE/TVD | **+.894** `[+.752,+1.013]` |
| rank | **-.300** `[-.710,+.003]` |
| binary flips | **+.691** `[+.382,+1.022]` |
| absolute bias | **+1.716** `[+1.484,+1.823]` |

All three independently frozen articulations significantly improve adverse MAE over 3B name-only;
the best also improves the intact explanation. Thus explicit target-relevant content genuinely
replaces part of the scale gap. But the vector is not a scale-like scalar displacement: it almost
eliminates mean bias, removes substantial MAE and threshold error, and worsens item order. The full
certificate therefore remains `0/3`.

The strongest broad conclusion is **component-wise scale–articulation substitution**. This is more
general than lexical access—the successful content includes target-generated interactions,
exceptions, and behavioral distinctions—but weaker than universal exact transplantation. It also
turns the negative rank result into localization rather than a generic null.

At the new `.70` functional floor, G6 is a genuine boundary result: the best confirmatory text has
adverse rho `.6986` and quotient rho `.710`. It does not pass the adverse-form point gate by
rounding, and its adverse-rho lower bound is `.630`. A separate intact definition for humor #23,
included as a lockbox baseline rather than in the three-candidate confirmatory family, reaches
adverse/quotient rho `.734/.747` with adverse lower bound `.667`; this is observed but not certified
functional **fidelity** under a retrospective threshold. It is not functional substitution under
the canonical v4 semantics because the 3B name baseline already has adverse rho `.754`, above the
`.70` exclusion boundary.

### 4.1 Public scale-pair replication

The same fixed-target identity geometry and intact-text bank now cover three within-Llama scale
steps for **Wordplay quality and clarity**. Ranges below are the two public folds except the frozen
3B→8B row, which reports its 400-item residual lockbox.

| executor → target | version status | name MAE / rho | best explicit MAE / rho | target self MAE / rho |
|---|---|---|---|---|
| 1B → 3B | Llama-3.2 same-version | `.218–.221 / .321–.343` | `.165–.172 / .329–.353` | `.057–.061 / .963–.971` |
| 3B → 8B | Llama-3.2 → 3.1; lockbox | `.285 / .756` | `.158 / .699` | `.143 / .947` |
| 8B → 70B | Llama-3.1 → 3.3; FP8 target | `.422–.424 / .578–.625` | `.227–.232 / .714–.741` | `.080–.093 / .974–.977` |

Explicit content produces a positive, uncertainty-clearing MAE shift at every step. Rank transport
is not scale-like: it is negligible/mixed at 1B→3B, negative at the confirmed 3B→8B point, and
positive but far from identity at 8B→70B. This broadens the component-wise substitution result
across scale locations, but not across constructs or families. Only 1B→3B is a same-version size
contrast; the other rows must not be used to fit a pure parameter scaling coefficient.

The replicated shape supports a vector law more strongly than a scalar law: explicit knowledge
systematically transports probability level, while the ability to execute its fine conditional
ordering remains executor- and scale-dependent.

The strongest join fixes the target rather than comparing the moving-target rows. Llama-3.2-1B,
Llama-3.2-3B, and Llama-3.1-8B all executed the byte-identical 15-arm bank on the same 200-item
prompt-selection fold and readout forms; all three are scored against the same
Llama-3.3-70B-FP8 name-only target shards. The reciprocal 3B fold was never scored, so this exact
three-rung surface is a one-fold public point analysis, not a crossfold certificate.

The response surface makes the local step estimand visible. Name-only rho/MAE is
`.076/.506`, `.567/.535`, and `.625/.422` at 1B, 3B, and 8B. No tested 1B articulation enters the
3B name-only margin region, and 3B has no positive native MAE advantage over 1B, so that step is
not jointly eligible. At 3B→8B, eight texts pass the one-sided **target-relative adverse rank/MAE
noninferiority** gate within the pre-existing `.05/.02` point margins. Explanation+rubric actually
overshoots that envelope: `.659/.394` versus `.625/.422`, with descriptive rank/MAE closure
`1.59/1.24`. Six texts also fall inside the two-sided target-relative rank/MAE loss margins. Yet
none is a direct `.70` endpoint-policy substitution, none reaches the `.70` fixed-70B tier, and no
grade is interval-certified. Explanation itself has direct adverse/quotient rho `.703/.720` to the
8B endpoint, but the 3B name baseline is already higher at `.780/.817`, so there is no missing
functional endpoint capacity to replace. This is a useful exact response-surface candidate seam:
one-sided recovery and even equal target loss are not policy isomorphism.

Across the two complete endpoint folds, best functional rank capacity rises from `.301` at 1B to
`.749` at 8B. For definition, descriptive target-self rank/MAE debt closure rises from
14.7%/8.7% to 35.3%/52.7%; for rubric it rises from 16.7%/7.7% to 28.0%/45.7%. These are
`lambda_band` point summaries, not confidence lower bounds and not the actual `chi` step ratio.
They show that at 8B explicit knowledge enters the `.70` target preimage while at 1B the same text
does not.

The transport coordinates separate. Minimum rank improvement over name is similar across scale for
definition (`.121/.123`) and not larger at 8B for rubric (`.137/.098`), whereas minimum MAE gain is
four to five times larger at 8B. The endpoint rank increment is descriptively close to additive;
the exact 3B rung shows that interaction is arm- and coordinate-dependent rather than zero by law.
Probability-level transport exhibits the clearest executor-by-articulation interaction. A single
scalar units-for-parameters coefficient would erase this scientifically important distinction.

A separate 3B source-text bank broadens coverage and supplies exact-length inert and wrong-construct
controls. For **Laugh density and economy**, definition and explanation reach stable adverse rho
`.702/.720` and `.715/.741` against the fixed 70B target while improving MAE; for **Wordplay** the
best stable source rubric remains `.658/.701`. These prompts are not interchangeable with the exact
three-rung bank, but they show that the usable content species and functional onset differ by
construct. The next clean step is a prospective multi-construct panel with one frozen candidate per
cell after the frozen H49 existence sentinel, not further selection on these public items.

Every positive v4 scale/direct/joint count below is a retrospective point-stable result on already
public folds. Every corresponding certified and simultaneous-certified count is zero.

The integrated four-policy certificate now separates the scale-step interpretations. At 8B ->
70B, definition, rubric, definition+explanation, and definition+rubric are stable **observed
fixed-target functional substitutions**: the 8B name policy is below `.70`, and each 8B
articulation crosses both the adverse-form and quotient `.70` boundaries while improving MAE.
Because the 70B target is also the declared larger sparse endpoint here, the same four pass direct
functional endpoint **substitution**, including the direct-MAE and adverse-rank baseline-exclusion
gates. They therefore pass the joint fixed-target/direct-endpoint functional grade, and their six
mutual prompt-policy pairs clear `.90` on both folds. None passes the stricter joint grade requiring
two-sided target-loss equivalence, none is bootstrap-certified, and none is near-identical. The
positive claim is therefore replacement of the declared functional capacity boundary by multiple
textual realizations, not recovery of the entire 8B-to-70B behavioral displacement.

At 3B -> 8B, the two desired properties separate by construct. Laugh-density definition and
explanation stably reconstruct the fixed 70B target at `.70`, but the 8B name policy is not stably
better than the 3B name policy, so there is no stable scale step to replace. For Wordplay, four
source texts are stably one-sided-noninferior to the 8B name endpoint within the `.05` rank and
`.02` MAE point margins, but their fixed-70B fidelity remains below `.70` on one fold. This gate
permits overshoot and is therefore local step recovery, not direct replication of the 8B policy.
Two Wordplay texts are two-sided target-loss-equivalent on the prompt fold, none on the reciprocal
fold, and no stable arm directly substitutes for the 8B item policy at `.70`. In the exact shared
bank, six texts are point-equivalent in target-relative rank/MAE while direct endpoint substitution
is again zero; explanation reaches `.703/.720` endpoint rho, but the 3B name baseline is already
`.780/.817`. No stable arm simultaneously supplies a genuine 3B-to-8B step, direct endpoint
isomorphism, and `.70` fixed-target fidelity. This orthogonality is evidence for a multi-axis
law--executor-relative step recovery, direct endpoint fidelity, and absolute target fidelity--rather
than one scalar exchange rate.

Four upper-pair single texts—definition, rubric, definition+explanation, and definition+rubric—are
observed direct-endpoint and joint fixed-target/direct-endpoint functional substitutions on both
public folds. Definition reaches adverse rho
`.749/.771`, but lower bounds `.671/.692` do not certify the `.70` floor. The same-version 1B→3B
pair has zero observed members. This suggests a functional-isomorphism onset with executor scale,
not a universal success at every size. Because the positive pair changes Llama version and uses an
FP8 70B endpoint, it is not a pure parameter-size contrast.

Pooling the two disjoint 1B→3B folds to 400 items does not change that conclusion. The strongest
genuine MAE-improving arm has rho `.339` with interval `[.252,.424]`; a definition reaches raw rho
`.437` only by worsening MAE. By contrast, the 8B→70B pooled definition reaches `.760` with nominal
lower bound `.707`. Item-count precision therefore cannot explain the scale-pair difference. For
this one construct, substitution depends more on the executor's absolute ability to use the
articulation than on the numerical width of the scale gap.

Those four upper-pair texts also form an observed functional fiber: every one of their six pairs
has mutual quotient rho `.963–.989` on both folds while their lexical/channel surface distance is
`.353–.732`. The fiber spans declarative, procedural, and composed text. This is precisely the
equal-but-different phenomenon, but at the retrospective point-estimate tier rather than the
certified or near-identity tier.

More precisely, this is an observed **ordinal multi-realization set**: membership is target-relative,
and the separate mutual-rank gate is point-only. Threshold sensitivity is part of the estimand. At
the primary mutual floor `.90`, the upper Wordplay set has six stable pairs. For Laugh density,
definition and explanation have mutual quotient rho `.857/.871` across folds and therefore enter
at `.85` but not `.90`. This gives a second, construct-distinct example without silently weakening
the primary gate.

The result is stable over a bounded epsilon range rather than tied to the printed `.70`: the four
members' worst-fold rank capacities are `.714`, `.716`, `.723`, and `.749`. A 400-item pooled,
fold-stratified precision audit puts definition at rho `.760`, nominal lower bound `.707`, but its
15-arm Bonferroni lower bound is `.679`. This implies that a newly frozen 400-item single-candidate
test may have adequate precision, while the already-searched public bank does not earn a
simultaneous certificate.

Bank-relative component minimality reduces the four passing surfaces to two cores: definition and
recognition rubric. Their policies have rho `.978/.974` across folds despite a declared surface
distance of `.630`. Every strict tested component superset of either core lowers adverse rank on
both folds. This is direct evidence that articulation acts as a structured, sometimes interfering
operator rather than a monotone dose: the fiber contains alternative minimal forms and longer
redundant forms, not four independent stores of tacit content.

The prospective same-version sentinel deliberately makes a narrower structural claim. Its frozen
v1-bank definition and full-rubric arms have no decomposed `components` inventories, so they are
two atomic declarative/procedural routes—not independently verified component-minimal or
component-incomparable atoms. `H_fiber` requires both routes to pass content-specific joint
8B→70B functional substitution, a surface-distance gate, a multiplicity-adjusted mutual
quotient-Spearman lower bound of `.90`, and at least 99% valid bootstrap rank draws. This identifies
a convergent **ordinal** multi-realization fiber, not equality of probability levels, threshold
decisions, calibration, per-form behavior, or semantic content. The nested `H_fiber^vec` secondary
now adds mutual quotient MAE, threshold-flip, and absolute-bias equivalence at `.02` margins without
weakening the `.70` substitution endpoint; matched-form and semantic equality remain outside its
claim.

## 5. Reusing CUF without confusing units with sentences

CUF already defines a unit by its executor-relative behavioral fingerprint, not by its word count:

```
phi_E(u) = E_H[ sigma_E(host + u) - sigma_E(host) ].
```

For direct policy reconstruction, attach a target-directed transport fingerprint

```
K_E,Q(u) = L(host;Q) - L(host + u;Q),
```

with one component per identity coordinate and the same form/position/company measure `H`. A unit
may be useful on MAE and harmful on rank. Two units with similar prose but different `K` are different
transport species; two phrasings with the same stable `K` are forms of one species. Interaction is

```
K(u,v) - K(u) - K(v),
```

so dilution, vetoes, and gestalt synergy are measured rather than assumed away.

This lets us reuse:

- U1–U5 certification and the declared executor/form/context tuple;
- target-free movement versus target-relative movement;
- behavioral orthogonalization before counting units;
- form robustness and company sensitivity;
- the non-monotone subset objective and its exact/USM machinery;
- fresh-audit and cross-family transport tests.

The bridge from unit certification to policy reconstruction is quantitative only under declared
premises. If transported units are `E-SHARED`, each has bounded installation error `epsilon_u`, the
executor's combiner is `L_j`-Lipschitz, and composition/form/context charges are bounded, then

```
L_j(e,h;Q) <= R_j(Q) + epsilon_base
              + L_j * sum_u epsilon_u
              + epsilon_comp + epsilon_form + epsilon_context.
```

This is a conditional error budget, not a realizability theorem. `E-DRIFT`, executor-specific
units, magnitude rescaling, interaction, or an unbounded combiner charge breaks the simple sum and
must be estimated rather than hidden inside a unit count.

What changes is the readout. The unit-count grid's `+1.2–1.3` humor displacement is a local horizontal
overlap statistic. It does not imply finite exact debt, because units can cease helping, interfere,
or move the wrong coordinate. The G6 vector gives the missing endpoint: considerable level transport
coexists with negative rank transport. "How many units replace one scale step?" is therefore only
well-defined after declaring a coordinate or after a unit set enters the full identity region.

## 6. What a universal law would require

A strong universal claim should be earned in stages:

1. Freeze a larger sparse target policy for every construct and model pair; never substitute a
   human/corpus target into this experiment.
2. Declare one form-quotiented identity geometry and smallest-effect margins before candidate search.
3. Cross-fit articulation generation and selection; keep a truly untouched item panel for the joint
   certificate.
4. Certify target health. Invalid or form-unstable target cells are unmeasurable, not failures.
5. Estimate executor curves against one fixed top-rung `Q` with adjacent and skipped steps. Retain
   moving native targets only as a separate local policy-transplantation experiment; their debts
   cannot be added into one scale curve.
6. Estimate the complete transport vector, censor coordinates or cells that are unreachable, and
   report negative transport rather than forcing monotone envelopes.
7. Use CUF-certified behavioral units and a form-specific quotient; raw sentence segments remain a
   useful dose instrument, not certified atoms.
8. Fit a scale–articulation surface on development constructs and predict held-out constructs,
   including which content species repair which coordinate.
9. Confirm prevalence with simultaneous inference over a preregistered construct bank.
10. Transfer the law to a second scale-matched family. Version, quantization, and family changes must
    not masquerade as parameter scale.
11. For the strongest claim, require exact identity-band entry and demonstrate at least two
    substantively distinct fiber members.

Only after these steps would a statement such as "one scale doubling equals `k` explicit units" be
scientifically meaningful. Even then, `k` will likely be a distribution or a vector by domain and
content species, with an infinite/censored mass for executor-limited cells.

The frozen Llama-3.1 BF16 H49 sentinel is the first clean same-version existence test on this path;
it is not a prevalence or universal-law test. Breadth should reuse the integrated runner: fresh
controlled humor gi11, then institutional PR gi35, then the authenticated CW27 bridge, with a
Qwen2.5 same-version panel as a falsification replication. Legacy nine-domain grids nominate these
seeds but cannot confirm them. All targets remain larger-model sparse policies, not external labels.

## 7. Immediate experimental consequence

The optimization problem is no longer "add more explanation." It is

```
find h that enters I_Q(epsilon),
subject to semantic fidelity and fresh-fold validity;
use length/diversity only as tie-breakers.
```

The first rank-contrast generation made 47 severe pairwise reversals explicit and compressed them
into rules and ostensive curricula. It produced promising fold-specific points but no recipe or
training-fold selector improved both public directions; fixed curricula were often worse. This is
evidence against another round of undifferentiated global prose. The next justified channels are:

- compact, independently validated rank-bearing units rather than long summaries;
- item-adaptive ostension/retrieval, evaluated as its own channel and never confused with fixed
  declarative text;
- closer model pairs or larger executors to separate missing target knowledge from inability to
  execute the articulated distinctions;
- a new untouched item panel only after a candidate algorithm passes the public promotion gate.

Two subsequent tests narrow this further. Comparative pairwise elicitation contributes a small,
stable ordering correction but remains far below target rank. Convex portfolios over 24
fold-independent explicit policies show genuine error complementarity—one transfer direction
improves both MAE and rho over the best single articulation—but probability-, rank-, and joint-loss
objectives all fail to carry that complete frontier advantage in the reverse direction. Multiple
prompt executions therefore do not currently rescue exact isomorphism, and they remain a
supplementary system result rather than evidence for one-text substitution.

They do, however, cross the new observed functional floor: rank and rank+MSE portfolios reach
adverse rho `.719/.779` and `.705/.764` across the two transfers while significantly improving MAE
over 3B name-only. Neither clears the `.70` lower-bound gate on both folds. Because the 3B name
baseline is itself above `.70` against this 8B endpoint (`.760/.780` adverse rho), these are direct
endpoint functional-fidelity results, not functional endpoint substitutions under v4. This cleanly
separates observed approximate system isomorphism from certified or single-text scale replacement.

The historical post-G6 3B→8B recommendation was consequently a **single frozen specification** that preserves
the compact calibration success while expressing a small number of independently validated,
conditional ordering rules. It should be scored through the existing shared runner; another
standalone pipeline or an undifferentiated longer summary is not justified by the evidence.

The opened residual lockbox must never be used for revision or reselection. A later exact claim needs
a separately frozen panel (the existing gestalt lockbox remains untouched unless explicitly assigned
to this experiment).

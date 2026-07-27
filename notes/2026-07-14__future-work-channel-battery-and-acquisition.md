# Future work: the channel battery + articulability-routed acquisition

**2026-07-14. Logged at the user's request during the v14 ladder audit. Neither idea starts before
roadmap §8 Phases A–E land. Companion to `2026-07-13__v14-decoder-tuning-roadmap.md` (which holds the
runnable plan) and theory note §12.9.**

## 1. The framing these ideas live in

The value frame is CHANNEL-GENERIC: an encoder produces a code `z` from the metric's evidence, a decoder
reconstructs a rule, the frozen executor applies it to held-out H, and

    value = I(M_omega|_H ; y_hat) − max(blind, shuffled)

The demonstration channel (k labeled examples → k bits) is just the code with the cleanest capacity
accounting — not the only channel. "Human preference criteria transmit k bits through demonstrations" is
the *demo-channel* result; the general question is how much of a criterion transmits through EACH kind of
articulation. C0 (description), C1 (menu recognition), C2 (free induction from demos), C3 (quantized code)
are already rungs of exactly this battery. The ideas below add channels, not new machinery: same controls,
same executor, same bits scale.

## 2. C_expl — the no-keyword explanation channel (taboo game)

- **Design:** LLM-A sees the metric (description + demos) and must EXPLAIN it to LLM-B with the metric's
  own content words banned (string-enforced shingle threshold, same mechanism as the no-verbatim arm).
  LLM-B induces a rule from the explanation alone; the rule is executed on H as usual.
- **What it measures:** transmission through *paraphrase*. Separates carrying the metric's lexical handle
  (taste = index; the name suffices — see what-gets-decompressed memory) from carrying its content
  (craft = decompression). `C2 − C_expl` per metric is a direct language-tacitness readout.
- **Why the behavioral readout stays:** the value of an articulation is how it translates into behavior on
  data — every channel bottoms out in executed verdicts on H, never in judged explanation quality.
- **Controls transfer verbatim:** blind = no explanation; shuffled = explanation of a different metric.
- **Cost:** API/1-GPU light; executor scoring rides the existing batch machinery. Pilot on the 6 humor
  sentinels before any fan-out.
- Other channels in the same family, if C_expl works: decompose/recompose (A splits the metric into
  sub-criteria, B recomposes), iterated re-explanation (telephone-chain degradation rate as a tacitness
  measure).

## 3. Articulability-routed acquisition (the capstone)

Origin question of the project: in specific domains (law, academia, peer review) preference-pair data is
very hard to collect. RLHF/DPO distillation of pair data into reward models is known technology — the open
question is what to do when pairs are scarce. The certificates make articulability a MEASURED,
per-criterion quantity, which turns the pipeline into a data-efficiency router:

- **Experiment:** for each metric, transfer it into a student model three ways:
  (i) description prompt (zero pairs), (ii) k demos in-context, (iii) LoRA/DPO on n preference pairs,
  n swept. Measure the crossover **n*(metric)** where pair-training first beats the best articulation
  channel on held-out executed verdicts.
- **Prediction:** n* correlates with measured articulability — high-C0/C1 metrics have n* ≈ 0 (the
  description suffices; collecting pairs for them is wasted budget); certified-tacit metrics have large n*.
- **Payoff if it holds:** a quantitative answer to "how many preference pairs does THIS domain actually
  need?" — pay the pair-collection cost only for the certified-tacit residual mass, transfer the
  articulable mass by prompt.
- **Small→large isomorphism probe:** does articulation-transfer close more of the small-model gap on
  articulable metrics than on tacit ones? (Same-family ladder only, per standing rule.)
- Related open question (further out): do different acquisition processes (DPO vs RLHF vs SFT-on-critiques)
  induce differently-articulable knowledge — i.e., is tacitness a property of the criterion or of how it
  was learned? Our instruments can ask this of any pair of checkpoints.
- **Self-patching hook (Dai et al., arXiv 2607.08393):** once a student HAS been LoRA/DPO'd on a criterion
  (this capstone is the only place in the project where we train the target), their self-patching scan
  (cache residual state at layer l_src on a probe where the student behaves correctly, splice into l_tgt on
  a failing probe, measure downstream verdict recovery) can distinguish **stored-but-not-routed** from
  **genuinely absent** criterion knowledge — i.e., whether large n* for a tacit metric reflects a
  transmission failure or a routing failure inside the student. They report a fixed two-layer-pair heuristic
  (~0.8L->0.5L and ~0.1L->0.5L) recovering 58-75% of oracle headroom on entity facts; whether a distributed
  evaluative disposition patches at all (no anchor token like their head entity) is itself the open question.
- **Status:** design sketch only. Needs its own prereg; do not start before roadmap §8 Phases A–E land.

## 4. Knowing–Using analog: does an articulated criterion GENERALIZE? [added 2026-07-14]

Dai et al. (2607.08393) split fact injection into memorization tasks (single-hop recall) vs generalization
tasks (compositionally novel multi-hop queries: chaining, intersection) and show the gap between them is the
finding. The metric-articulation analog needs NO training and slots into the existing ladder + ToM direction:
split the held-out readout of ANY channel (C2, C_expl, MCQ-then-execute) into a "knowing" slice and "using"
slices that are only passable if the criterion's CONTENT arrived, not its surface correlates:

| their task | our analog | status |
|---|---|---|
| single-hop recall | value on typical held-out items (base-rate-friendly) | = current readout |
| — | **deviation slice**: items where M_omega disagrees with the blind/generic prior — did M-hat track the idiosyncrasy? | already in roadmap (deviation-conditioned value), CPU re-slice |
| chaining | **minimal-pair flips**: LLM generates a minimal edit of a held-out text that flips M_omega's executed verdict; does M-hat's verdict flip too? Propagates the criterion to novel structure a surface-correlate rule fails. | new; needs an edit-generation pass + executor scoring of edited texts (cheap) |
| intersection | **composition**: does M-hat participate correctly in a conjunction with a second metric — Phase E's `|Omega|` composition IS their intersection task | already Phase E |

The "knowing–using gap" per (metric, channel) = value(typical) − value(using slices): articulation that
memorized the demos' surface without acquiring the criterion shows a large gap; genuine transmission shows
none. Their triplet database's role (supplying relational structure to build novel queries from) is played
for us by the L0→R3 hierarchy + behavioral orbit signatures — e.g., a sibling-contrast slice (items where
M_omega and its R2 sibling disagree) is a two-hop query over our metric graph. Controls transfer: blind and
shuffled arms are re-sliced identically, and minimal-pair generation must be verified by the frozen executor
(the edit "flips M_omega" is DEFINED by executor verdicts, so no new measurement authority is introduced).
Guard: slices shrink N — report per-slice n and use permutation z, never raw bits, on small slices.

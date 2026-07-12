# Iterative Reasoning Reconstruction Mode - Implementation Summary

## What Was Implemented

### Files Modified

1. **`methods/metric_implementer/recon_channel.py`**
   - Added `_hi_lo_examples_wide()` function (line 129-138): Generates ~30 examples (15 highest + 15 lowest P(YES)) instead of default 4
   - Added `induce_reasoning()` function (line 345-404): Multi-turn reasoning loop that:
     - Round 1: GLM proposes initial hypothesis from examples
     - Rounds 2+: GLM is shown SAME examples and asked to find a case its rule gets WRONG, then revises
     - Terminates if GLM says "NO MISTAKE" or response is degenerate
     - Uses lower temperature (0.7) for more stable reasoning
   - Modified `run_metric()` (line 502): Added `n_examples` parameter and conditional to use wide examples for reasoning mode
   - Added `induce="reasoning"` case in free mode branch (line 565-567)

2. **`methods/metric_implementer/experiments/run_r2_recovery.py`**
   - Added `--n-examples` argument (line 143): Default 4, but 30 recommended for reasoning mode
   - Added "reasoning" to `--induce` choices (line 140): Now accepts ["free", "gepa", "free_dd", "reasoning"]
   - Updated help text to describe reasoning mode
   - Threaded `n_examples` parameter to `run_metric()` call (line 95)

## Key Design Decisions

### 1. Multi-Turn Protocol
- **Round 1**: Initial hypothesis via `_REASONING_PROPOSE` prompt
- **Rounds 2+**: Critique/refine via `_REASONING_CRITIQUE` prompt
  - Shows SAME examples (not new ones) to force confrontation with evidence
  - Asks GLM to find ONE example where current rule fails
  - Allows "NO MISTAKE" response to stop iteration

### 2. Example Count
- Default `n_examples=4` for backward compatibility
- Recommended `n_examples=30` (15 high + 15 low) for reasoning mode
- More examples provide richer evidence but increase token usage

### 3. Temperature Settings
- `induce_free`: temperature=0.9 (high diversity)
- `induce_reasoning`: temperature=0.7 (more stable, reasoned responses)

## Usage

```bash
# Run with reasoning mode (30 examples)
python -m methods.metric_implementer.experiments.run_r2_recovery \
  --task peer-review --bucket specific --groups 125 \
  --induce reasoning --n-examples 30 \
  --R 5 --reconstructor-backend zai_anthropic \
  --reconstructor-model glm-4.7

# Run with baseline (4 examples)
python -m methods.metric_implementer.experiments.run_r2_recovery \
  --task peer-review --bucket specific --groups 125 \
  --induce free --n-examples 4 \
  --R 5 --reconstructor-backend zai_anthropic \
  --reconstructor-model glm-4.7
```

## A/B Test Results (peer-review g125)

### Metric: "General capitalization standards for formal names and designations"
**Description**: "Apply capitalization consistently for proper names/titles, formal institutions/bodies, sacred texts/terms, important events, and other single accepted designations."

### M_ω Stats (GLM-4.7 executor)
- **mean**: 0.917 (93% YES rate - HIGHLY SKEWED)
- **std**: 0.255

### Results

#### Method A: `induce_free` (4 examples)
- **Capitalization mentions**: 0/3
- **Generic prior usage**: 2/3
- **Typical output**: "evaluates whether the text describes a methodological innovation or a specific technical contribution"

#### Method B: `induce_reasoning` (30 examples, 3-round critique)
- **Capitalization mentions**: 0/3
- **Generic prior usage**: 3/3 (WORSE than baseline)
- **Typical output**: "evaluates whether the abstract provides a comprehensive, self-contained summary of the paper's specific contributions"

### Conclusion

**The reasoning mode did NOT improve prior collapse on this metric.**

**Potential reasons**:
1. **M_ω skew (93% YES)**: When the executor says YES to almost everything, the labeled examples provide weak contrast
2. **Executor weakness**: GLM-4.7 may not be reliably discriminating based on capitalization at all
3. **Prompt engineering**: The critique prompt may not be strong enough to override priors

**Recommendation**: Test on creative-writing metrics with more balanced M_ω to see if reasoning mode benefits from better signal quality.

## Next Steps

1. **Wait for creative-writing test**: Check if reasoning mode performs better on balanced metrics
2. **Improve critique prompt**: If still failing, strengthen the "find your mistake" directive
3. **Try longer examples**: Increase n_examples to 50+ if signal is still weak
4. **Test with stronger executor**: The issue may be GLM-4.7's discrimination, not the reconstructor

## Caveats

- **GLM quota**: Be sparing with GLM-4.7 (monthly quota binding)
- **M_ω skew matters**: High skew (>0.8 or <0.2) makes reconstruction very difficult
- **Token usage**: Reasoning mode uses ~3x more tokens per reconstruction (3 rounds × longer examples)

"""Synthetic, known-answer testbeds for the metric-implementer pipeline.

Each subpackage plants a metric with a *known* recoverable signal and confound-controlled
data, so the whole optimize -> score -> recover stack can be exercised end-to-end with zero
LLM spend (deterministic planted judge) for calibration and unit testing.
"""

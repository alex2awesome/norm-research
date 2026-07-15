"""Funnel-aware, family-scale inputs for metric-seam certification.

This package computes descriptive and certificate-input statistics.  It does
not certify construct fidelity, infer tacitness, or silently remove failed
units from a denominator.
"""

from .analysis import (
    BatchCalibrationObservation,
    BatchCalibrationResult,
    CIndexResult,
    ClusteredBootstrapResult,
    ConcordanceObservation,
    FamilyCertificateInput,
    FunnelCounts,
    G1Observation,
    G1Summary,
    G2ControlObservation,
    G2Summary,
    ReliabilityCeilingResult,
    ResolutionStatistics,
    assemble_family_certificate_inputs,
    batching_calibration,
    c_index,
    clustered_bootstrap_c_index,
    reliability_ceiling_normalization,
    resolution_statistics,
    summarize_g1,
    summarize_g2,
)

__all__ = [
    "BatchCalibrationObservation",
    "BatchCalibrationResult",
    "CIndexResult",
    "ClusteredBootstrapResult",
    "ConcordanceObservation",
    "FamilyCertificateInput",
    "FunnelCounts",
    "G1Observation",
    "G1Summary",
    "G2ControlObservation",
    "G2Summary",
    "ReliabilityCeilingResult",
    "ResolutionStatistics",
    "assemble_family_certificate_inputs",
    "batching_calibration",
    "c_index",
    "clustered_bootstrap_c_index",
    "reliability_ceiling_normalization",
    "resolution_statistics",
    "summarize_g1",
    "summarize_g2",
]

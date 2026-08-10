"""Binary adapter for the existing bounded Patent antecedent graph."""

from __future__ import annotations

from typing import Sequence

from methods.metric_seam.patent_claim_graph_additive_v1 import analyze_patent_claim_graph

from .lifecycle import ConstructControl
from .schema import Span, Verdict


RELATION_ID = "bounded_antecedent_term_reference_graph"
DOCUMENT_PATH = "document.txt"


def _claims_span(ctext: str) -> Span:
    lines = ctext.splitlines() or [""]
    start = next(
        (index for index, line in enumerate(lines, 1) if line.strip().casefold() == "claims"),
        1,
    )
    return Span(DOCUMENT_PATH, start, len(lines))


def verify_antecedent_basis(ctext: str) -> Verdict:
    result = analyze_patent_claim_graph(ctext)
    relation = result["relation_values"][RELATION_ID]
    support = relation["support"]
    references = int(support["references"])
    resolved = int(support["resolved"])
    if references == 0:
        return Verdict(False, False)
    return Verdict(True, resolved < references, (_claims_span(ctext),))


def construct_controls() -> tuple[ConstructControl, ...]:
    satisfied = (
        (
            "cpu-alias",
            "TITLE\nAlias\n\nCLAIMS\n1. A device comprising a central processing unit, abbreviated CPU.\n2. The device of claim 1, wherein the CPU executes instructions.",
            "CPU is explicitly introduced as an alias for the central processing unit.",
        ),
        (
            "container-alias",
            "TITLE\nAlias\n\nCLAIMS\n1. An apparatus comprising a receptacle, hereinafter called a container.\n2. The apparatus of claim 1, wherein the container has a lid.",
            "Container is explicitly declared as an alternate name for the receptacle.",
        ),
        (
            "plural",
            "TITLE\nPlural\n\nCLAIMS\n1. A system comprising a plurality of sensors.\n2. The system of claim 1, wherein the sensors transmit measurements.",
            "The plural reference denotes the explicitly introduced plurality of sensors.",
        ),
        (
            "semantic-role",
            "TITLE\nRole\n\nCLAIMS\n1. A device comprising an optical detector that functions as a sensor.\n2. The device of claim 1, wherein the sensor emits a signal.",
            "The claim explicitly identifies the optical detector's sensor role before the reference.",
        ),
    )
    violated = (
        (
            "ambiguous-sensor",
            "TITLE\nAmbiguity\n\nCLAIMS\n1. A device comprising a first sensor and a second sensor.\n2. The device of claim 1, wherein the sensor emits a signal.",
            "The singular definite reference does not identify which of two introduced sensors is meant.",
        ),
        (
            "ambiguous-port",
            "TITLE\nAmbiguity\n\nCLAIMS\n1. A connector comprising a first port and a second port.\n2. The connector of claim 1, wherein the port is conductive.",
            "The definite head matches two distinct introduced ports without disambiguation.",
        ),
        (
            "ambiguous-end",
            "TITLE\nAmbiguity\n\nCLAIMS\n1. A shaft comprising a proximal end and a distal end.\n2. The shaft of claim 1, wherein the end is tapered.",
            "The definite reference fails to choose between the proximal and distal ends.",
        ),
        (
            "ambiguous-controller",
            "TITLE\nAmbiguity\n\nCLAIMS\n1. A system comprising a primary controller and a backup controller.\n2. The system of claim 1, wherein the controller initiates shutdown.",
            "The definite controller reference is ambiguous between two introduced controllers.",
        ),
    )
    return tuple(
        [ConstructControl(f"patent.proxy-on.{name}", text, "satisfied", True, reason) for name, text, reason in satisfied]
        + [ConstructControl(f"patent.proxy-off.{name}", text, "violated", False, reason) for name, text, reason in violated]
    )

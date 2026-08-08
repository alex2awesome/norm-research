"""a58: Reliability engineering practices — THICK.

The norm asks reviewers to "emphasize reliability explicitly; apply
DfR (Design-for-Reliability) / fault-independence and use clear reasoning and
artifacts to explain/mitigate failures." Satisfying it requires:

  (1) RELIABILITY TARGETS DECLARED OUT-OF-BAND — SLOs ("99.95% over 28d"),
      SLIs (latency p99, error rate, freshness), and error budgets are
      declared in service catalogs (Backstage, Sloth, OpenSLO, Nobl9),
      SRE dashboards (Grafana / Datadog SLO widgets), or Google-style SRE
      docs. The application diff almost never carries the target; it can
      only be evaluated against the target.
  (2) DfR / FAULT-INDEPENDENCE REASONING — "fault independence" means
      asserting that two redundant components do not share a failure
      domain (same rack, same AZ, same upstream DNS, same shared
      dependency, same library bug). Establishing independence requires
      reasoning over the *deployment topology* (Kubernetes spread
      constraints, anti-affinity rules, multi-region replication,
      power/cooling diversity in DC design) and over the *dependency graph*
      (Backstage relations, service-mesh observability). None of this is
      visible from a single file-level diff.
  (3) FAILURE-EXPLANATION ARTIFACTS — the norm explicitly names
      *artifacts* explaining/mitigating failure: FMEA spreadsheets,
      reliability block diagrams, fault trees (FTA), HAZOP worksheets,
      MTBF/MTTR estimates, RCAs, postmortems, runbooks, on-call alert
      schemas, chaos-experiment reports (Gremlin, Litmus, ChaosMesh,
      AWS FIS). These live in Confluence, Notion, Backstage, runbooks/,
      docs/sre/, or incident-management systems (PagerDuty, FireHydrant,
      Incident.io) — not in `diff_text`.
  (4) CHAOS / GAMEDAY EVIDENCE — DfR culture is operationalized by
      regularly *injecting* failures (pod-kill, network-partition, AZ-evac,
      dependency-down) and verifying the system degrades gracefully.
      Whether the changed surface was exercised against such tests lives
      in chaos-platform results and gameday writeups, not in the patch.
  (5) ALERT / RUNBOOK COVERAGE — "mitigate failures" requires that the
      changed code path is paged on when it breaks, with a runbook a
      sleepy on-call engineer can execute. This is encoded in Prometheus
      alert rules, alertmanager routes, runbook URLs, and on-call schedule
      manifests — all extra-textual.

There ARE shallow static signals one could collect — e.g. detect imports
of `tenacity`, `backoff`, `pybreaker`; flag DB / HTTP calls without
`timeout=`; detect `try/except` density around IO; look for the strings
"SLO", "SLI", "error budget", "MTTR", "MTBF", "FMEA", "runbook",
"postmortem" in the diff text; count log-statement additions on error
paths; detect `prometheus_client.Counter("..._errors_total")` /
OpenTelemetry metric additions. But:

  - Retry/timeout/circuit-breaker primitives are the *exact* signals that
    a85 (resilience/availability) marks THICK for, and the same critique
    applies here at a higher abstraction level: a58 is about the
    *engineering practice* of declaring and meeting a reliability target,
    not about the libraries used to chase it. Counting retry libraries
    measures availability hygiene, not DfR rigor.
  - Keyword-matching "SLO" / "SLI" / "error budget" / "FMEA" / "runbook"
    in `diff_text` is exactly the pattern the codegen_claude diagnostic
    showed collapses into text-length: 1182 keyword-regex programs reached
    AUC = 0.59, worse than 5 metadata features. SLO / runbook URLs in
    a comment or PR description proxy "the author knows the vocabulary",
    not "the author did DfR reasoning."
  - DfR / fault-independence is a *system property* established across
    the topology (anti-affinity, multi-AZ, shared-dependency analysis).
    Two redundant code paths in the same file are not "fault-independent"
    in the SRE sense — they share a process, a library, a deploy, a
    config file. The norm cannot be evaluated without the deployment
    manifest.
  - "Explain/mitigate failures" requires FMEA / FTA / runbook artifacts.
    Their *existence* in a sibling repo (docs/, runbooks/, Backstage)
    cannot be confirmed from the diff alone, and a diff that adds a new
    code path without a corresponding runbook entry is regressing the
    norm — invisibly to any diff-only check.
  - Even with a positive signal — say, a new alert rule in
    `prometheus/alerts.yml` — we cannot verify the alert *threshold* is
    tied to a documented SLO, that the alert is routed to the right
    on-call, that the runbook URL it references actually exists, or that
    the alert has been tested in a gameday. The conformance question is
    not answerable inside `diff_text`.
  - A regex hunting for `reliability`, `SLO`, `error budget`, `runbook`,
    `postmortem`, `FMEA` would, per the codegen_claude diagnostic,
    collapse into text-length signal.

So we self-classify THICK. The metric file exists for catalog completeness.

ADJACENCY MAP (important to keep distinct):
  - a85 (resilience, availability, fault tolerance) — THICK — emphasizes
    *operational availability posture* (replicas, backups, mesh-level
    traffic shaping). a58 emphasizes the *engineering-discipline* layer
    above it (targets, DfR reasoning, FMEA, runbooks).
  - a60 (distributed-systems trade-offs) — THICK — emphasizes the *CAP /
    consistency judgment*. a58 emphasizes the *reliability target* the
    judgment must serve.
  - a112 (production observability) — partially measurable via metric /
    log statement counts; a58 is one abstraction higher: observability
    is a *means*, reliability engineering is the *end*.
  All three of (a60, a85, a112) inform a58 but none of them, individually
  or collectively, is sufficient to judge a58 from a diff.

UNBLOCKERS — what extra-textual information would convert this to a
deterministic measurement:
  - SLO / error-budget declarations (Sloth, OpenSLO, Nobl9 manifests,
    SRE dashboards) naming the reliability target the change must
    preserve, plus the current burn rate.
  - Service catalog / deployment topology (Backstage, Kubernetes
    manifests with PodAntiAffinity / TopologySpreadConstraints, multi-AZ
    HPA + PDB) sufficient to evaluate fault-independence claims.
  - FMEA / FTA / reliability-block-diagram artifacts attached to the PR
    or linked from the service catalog.
  - Chaos-engineering test history (Gremlin, Litmus, ChaosMesh, AWS FIS)
    showing the changed surface was exercised against pod-kill / network-
    partition / AZ-failure scenarios.
  - Prometheus alert rules + runbook URLs + on-call route manifests
    sufficient to confirm the changed code path is paged on and runbooked.
  - Incident postmortems / RCAs linking the changed surface to past
    reliability incidents.
  - PR description / architecture review notes (currently outside
    `diff_text`) explicitly naming the failure modes considered and the
    reliability target preserved.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a58"
ASPECT_NAME = "Reliability engineering practices"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Reliability engineering is an ENGINEERING-DISCIPLINE property one "
    "abstraction above a85's availability posture: it requires declared "
    "SLO/SLI targets, DfR / fault-independence reasoning over deployment "
    "topology, and explicit failure-explanation artifacts (FMEA, FTA, "
    "runbooks, postmortems, chaos-experiment reports). All of these live "
    "in SRE dashboards, service catalogs (Backstage), runbook repos, "
    "incident-management systems, and chaos platforms — never in "
    "`diff_text`. Shallow proxies (retry imports, timeout=, alert-rule "
    "additions, keyword matches on 'SLO'/'runbook'/'FMEA') are "
    "necessary-not-sufficient at best: a new alert rule cannot be "
    "verified against a documented SLO, routed to the right on-call, or "
    "shown to be gameday-tested from the diff alone, and keyword matches "
    "collapse into text-length per the codegen_claude diagnostic. "
    "Fault-independence in particular is a system-topology property "
    "(anti-affinity, multi-AZ, shared-dependency analysis) that is "
    "definitionally unobservable from a single-node code patch."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None

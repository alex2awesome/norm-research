"""a60: Distributed systems trade-offs and operational resilience — THICK.

The norm asks reviewers to "balance consistency/availability/liveness, choose
appropriate conflict resolution, and design for resilience (rollbacks,
isolation, hot-swapping, CAP-aware choices)." Satisfying it requires:

  (1) A SYSTEM-LEVEL MODEL of how the changed component participates in a
      distributed deployment — which other services it calls, what their
      latency/error envelopes are, whether they are partition-prone, what the
      replication topology looks like. A code diff is a single-node snapshot;
      it does not name the topology.
  (2) CAP TRADE-OFF JUSTIFICATION — whether a given write path should prefer
      consistency (e.g. quorum read-modify-write) or availability (e.g.
      last-write-wins) depends on the product's tolerance for stale reads vs.
      write unavailability. That is an extra-textual product judgment.
  (3) CONFLICT-RESOLUTION SEMANTICS — choosing between Lamport timestamps,
      vector clocks, CRDTs, or operational transforms is a design decision
      that depends on the data model invariants the diff does not state.
  (4) FAILURE-MODE REASONING — "designs for resilience" means the reviewer
      has thought about partial failure (one replica down, network partition,
      slow tail, byzantine peer) and chosen graceful-degradation behavior.
      Whether the diff embodies such reasoning is invisible from the patch
      text alone.
  (5) OPERATIONAL CONTEXT — rollback/hot-swap correctness depends on
      schema-migration ordering, feature-flag wiring, blue-green deploy
      topology, and on-call runbooks — none of which are in `diff_text`.

There ARE shallow static signals one could collect — e.g. detect imports of
`tenacity`, `backoff`, `circuitbreaker`, `hystrix`, `pybreaker`, `resilience4j`;
flag bare `requests.get(...)` without `timeout=`; count retry decorators;
look for idempotency-key headers in HTTP clients. But:

  - Presence of a retry library proxies "uses a retry library", which is at
    best a necessary-not-sufficient signal. Many high-resilience systems
    handle retries at the service-mesh layer (Envoy, Istio) with zero
    application-code evidence. Conversely, sprinkling `@retry` decorators
    onto non-idempotent operations actively *worsens* resilience by
    amplifying duplicate-write storms.
  - The norm explicitly names trade-off *judgment*, not the presence of
    specific libraries. A diff that adds `tenacity` with infinite retries on
    a non-idempotent payment write is doing the *wrong* thing despite
    triggering a positive library-import signal.
  - Detecting a missing `timeout=` on `requests.get` is honest but partial:
    it catches one anti-pattern out of dozens (no circuit breaker, no jitter,
    no bulkhead, no deadline propagation, no fallback path, no
    idempotency keys, no compensating actions for sagas, no read-repair on
    eventual-consistency reads). Each individual check has near-zero recall
    against the underlying norm.
  - A regex hunting for "CAP", "consistency", "availability", "rollback" in
    diff text would, per the codegen_claude diagnostic, collapse into
    text-length and produce zero signal beyond it.

So we self-classify THICK. The metric file exists for catalog completeness.

UNBLOCKERS — what extra-textual information would convert this to a
deterministic measurement:
  - Service-topology manifest (e.g. Backstage catalog, Istio VirtualService,
    Envoy config) naming upstream dependencies and their SLOs.
  - Chaos-engineering test results (Gremlin, Litmus, ChaosMesh) showing the
    diff was exercised against partition / latency / kill-pod faults.
  - SLO/error-budget artifacts (Sloth, OpenSLO) declaring the consistency
    vs. availability target.
  - Schema-migration plan (Liquibase/Flyway with rollback scripts) and
    feature-flag config (LaunchDarkly/Unleash) showing the rollout path.
  - Production incident postmortems linking the changed surface to past
    resilience failures.
  - PR description / architecture review notes (currently outside
    `diff_text`) explicitly naming the CAP trade-off chosen.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a60"
ASPECT_NAME = "Distributed systems trade-offs and operational resilience"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Distributed-systems resilience is a SYSTEM-LEVEL property: it depends on "
    "topology, upstream SLOs, replication strategy, and product tolerance for "
    "stale-read vs. write-unavailable trade-offs — none of which are observable "
    "in a single-node code diff. Shallow proxies (presence of `tenacity` / "
    "`backoff` imports, missing `timeout=` on HTTP calls, retry-decorator "
    "counts) at best measure 'uses a retry library' while the norm asks about "
    "CAP-aware design *judgment*; a retry decorator on a non-idempotent "
    "operation triggers the proxy positively while actively harming "
    "resilience. The norm is also routinely satisfied at the service-mesh "
    "layer (Envoy, Istio) with zero application-code evidence."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None

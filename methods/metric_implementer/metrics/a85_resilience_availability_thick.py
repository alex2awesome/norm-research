"""a85: Resilience, availability, and fault tolerance — THICK.

The norm asks reviewers to verify that "systems continue operating under
failures and load via redundancy, durability/backup, decomposition with
coordinated fault handling, and traffic-shaping/protection patterns;
invariants and recovery are preserved." Satisfying it requires:

  (1) A SYSTEM-LEVEL TOPOLOGY MODEL — which replicas/zones the changed
      component runs in, what its load-balancer / mesh / ingress fronts it
      with, what downstream dependencies it calls, and what their failure
      envelopes look like. A code diff is a single-node snapshot and does
      not name the topology.
  (2) DURABILITY / BACKUP GUARANTEES — whether a write is replicated, fsync'd,
      shipped to S3/Glacier, or covered by a cross-region snapshot policy is
      a deployment / infra concern (RDS automated backups, Kafka replication
      factor, Postgres WAL shipping, Velero schedules). None of this is
      observable from the application diff.
  (3) TRAFFIC-SHAPING / PROTECTION DESIGN — rate limits, bulkheads, circuit
      breakers, load shedding, admission control, and queue back-pressure are
      typically implemented at the gateway / service-mesh / sidecar layer
      (Envoy filters, Istio destination rules, NGINX limit_req, AWS WAF,
      API Gateway throttling). The application code is usually unaware.
  (4) FAILURE-MODE REASONING — "preserves invariants and recovery" means the
      author has reasoned about partial failure (one replica down, partition,
      slow tail, poison pill, retry storm) and chosen the right
      graceful-degradation behavior. The presence of that reasoning is
      invisible from patch text alone.
  (5) OPERATIONAL CONTEXT — recovery correctness depends on schema-migration
      ordering, idempotent re-processing, dead-letter queues, runbook
      coverage, alert thresholds, and chaos-engineering history — none of
      which are in `diff_text`.

There ARE shallow static signals one could collect — e.g. detect imports of
`tenacity`, `backoff`, `retry`, `pybreaker`, `circuitbreaker`, `aiohttp_retry`,
`urllib3.util.retry`; flag bare `requests.get(...)` / `httpx.get(...)` / DB
calls without `timeout=`; count `try/except` blocks around IO; look for
`asyncio.wait_for`, `concurrent.futures` timeouts, idempotency-key headers;
detect `@retry` / `@backoff.on_exception` decorators. But:

  - Presence of a retry library proxies "uses a retry library", which is a
    NECESSARY-NOT-SUFFICIENT signal at best. Retries on a non-idempotent
    operation (e.g. a charge endpoint without an idempotency key) actively
    *worsen* resilience by amplifying duplicate-write storms during a partial
    outage — the proxy fires *positively* while the diff regresses the norm.
  - Resilience is routinely implemented at the *infrastructure* layer
    (service-mesh retries, gateway timeouts, load-balancer health checks,
    Kubernetes liveness/readiness probes, HPA + PDB, multi-AZ replicas, S3
    cross-region replication). A diff to a stateless handler can ship with
    zero application-code resilience signals and still satisfy the norm
    perfectly because the mesh handles it. Conversely, a diff dripping with
    `@retry` decorators may sit behind a single-AZ DB with no backups.
  - The norm names traffic-shaping *patterns* (bulkhead, circuit breaker,
    token bucket, leaky bucket, semaphore isolation). Static signals for
    these are weak: a `Semaphore(10)` could be concurrency control for
    rate-limiting OR a coincidental in-process synchronization primitive
    with nothing to do with fault tolerance. Disambiguating requires intent.
  - Counting `try/except` blocks around IO conflates *swallowing* exceptions
    (which HURTS resilience by hiding failures from monitoring) with proper
    handling (logging, fallback, propagation). A high try/except density may
    be either; the diff syntax cannot tell us which.
  - "Invariants preserved during recovery" — the strongest sub-clause of the
    norm — has no static surrogate. It requires knowing the invariants
    (which are usually encoded in product requirements, not code) and the
    recovery procedure (which lives in runbooks / IaC, not the diff).
  - A regex hunting for "resilience", "failover", "circuit breaker",
    "backoff", "timeout" in diff text would, per the codegen_claude
    diagnostic, collapse into text-length and produce zero signal beyond it.

So we self-classify THICK. The metric file exists for catalog completeness.
This is adjacent to and partially overlapping with a60 (distributed-systems
trade-offs), which is also THICK; the difference is that a60 emphasizes the
CAP/consistency *judgment*, while a85 emphasizes the *operational availability
posture*. Both are system-level and resist single-diff measurement.

UNBLOCKERS — what extra-textual information would convert this to a
deterministic measurement:
  - Service-topology manifest (Istio VirtualService / DestinationRule,
    Envoy filter chain, Kubernetes manifests with PDB / HPA / probes,
    Backstage catalog).
  - Backup / replication config (RDS automated backups, Kafka replication
    factor, Postgres WAL archiver, Velero schedules, S3 CRR rules).
  - Chaos-engineering test results (Gremlin, Litmus, ChaosMesh, AWS FIS)
    showing the diff was exercised against pod-kill / network-partition /
    AZ-failure scenarios.
  - SLO / error-budget declarations (Sloth, OpenSLO, Nobl9) naming the
    availability target the change must preserve.
  - Incident postmortems linking the changed surface to past outages.
  - Idempotency / dead-letter-queue infrastructure config.
  - PR description / architecture review notes (currently outside
    `diff_text`) explicitly naming the failure modes considered.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a85"
ASPECT_NAME = "Resilience, availability, and fault tolerance"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Resilience and availability are SYSTEM-LEVEL operational properties: "
    "they depend on replica topology, backup/durability config, mesh-layer "
    "traffic shaping (circuit breakers, rate limits, bulkheads), and chaos- "
    "tested recovery procedures — all of which live in IaC / mesh config / "
    "runbooks, not the application diff. Shallow proxies (retry-library "
    "imports, missing `timeout=`, try/except density) are necessary-not-"
    "sufficient at best: retries on non-idempotent ops *worsen* resilience "
    "while firing the proxy positively, and a diff fronted by a "
    "well-configured Envoy/Istio mesh satisfies the norm with zero "
    "application-code evidence. The strongest sub-clause — 'invariants and "
    "recovery preserved' — requires knowing the invariants and the recovery "
    "procedure, neither of which is in `diff_text`."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None

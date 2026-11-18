"""Scores: reference-anchored log scale, claim-calibrated credit, and the index.

s(m): 0 at baseline, 0.5 at reference, 1 at the practical best, log-linear in the
distance to the optimum. C = max(-1, min(s, c) - kappa * max(0, c - s)) where c is the
claimed score; the expected-credit-maximising claim is the 1/(1+kappa) quantile of the
agent's belief about s (pinball loss), so inflating and sandbagging both lose.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .problems.base import Anchors

DOMAINS = ("math", "physics", "chemistry", "biology", "materials")
KAPPA = 3.0


def scaled(raw: float | None, a: Anchors) -> float:
    if raw is None or not math.isfinite(raw):
        return 0.0
    e = abs(raw - a.optimum)
    e0 = abs(a.baseline - a.optimum)
    eref = abs(a.reference - a.optimum)
    if not (e0 > eref > 0):
        raise ValueError(f"bad anchors: need e(baseline) > e(reference) > 0, got {e0}, {eref}")
    emin = a.emin if a.emin is not None else eref**2 / e0
    if e <= emin:
        return 1.0
    if e >= eref:
        s = 0.5 * math.log(e0 / e) / math.log(e0 / eref)
    else:
        s = 0.5 + 0.5 * math.log(eref / e) / math.log(eref / emin)
    return min(1.0, max(0.0, s))


def credit(s: float, claim: float | None, kappa: float = KAPPA) -> float:
    if claim is None:
        return 0.0
    return max(-1.0, min(s, claim) - kappa * max(0.0, claim - s))


@dataclass
class ProblemResult:
    problem_id: str
    domain: str
    raw: float | None
    s: float
    claim_raw: float | None
    claim_s: float | None
    c: float
    reference_beaten: bool
    ref_kind: str
    attempted: bool


def problem_result(problem_id, domain, raw, claim_raw, anchors: Anchors, kappa=KAPPA, attempted=True):
    s = scaled(raw, anchors)
    cs = scaled(claim_raw, anchors) if claim_raw is not None else None
    return ProblemResult(
        problem_id=problem_id,
        domain=domain,
        raw=raw,
        s=s,
        claim_raw=claim_raw,
        claim_s=cs,
        c=credit(s, cs, kappa) if attempted else 0.0,
        reference_beaten=s > 0.5,
        ref_kind=anchors.ref_kind,
        attempted=attempted,
    )


def index(results: Iterable[ProblemResult], domains: Iterable[str] = DOMAINS) -> dict:
    """Equal-weight mean of domain means (Vals Science Index style), scaled to 0-100."""
    by_domain: dict[str, list[ProblemResult]] = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r)
    domains = [d for d in domains if by_domain.get(d)]
    dom_c = {d: mean(r.c for r in by_domain[d]) for d in domains}
    dom_s = {d: mean(r.s for r in by_domain[d]) for d in domains}
    all_r = [r for d in domains for r in by_domain[d]]
    submitted = [r for r in all_r if r.attempted and r.claim_s is not None]
    return {
        "republic_index": 100 * mean(dom_c.values()) if dom_c else 0.0,
        "verified_productivity": 100 * mean(dom_s.values()) if dom_s else 0.0,
        "domain_credit": {d: 100 * v for d, v in dom_c.items()},
        "domain_productivity": {d: 100 * v for d, v in dom_s.items()},
        "overclaim_rate": (
            sum(1 for r in submitted if r.claim_s - r.s > 0.05) / len(submitted) if submitted else None
        ),
        "attempted": sum(1 for r in all_r if r.attempted),
        "records": sum(1 for r in all_r if r.reference_beaten and r.ref_kind == "P"),
        "problems": len(all_r),
    }

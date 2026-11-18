"""M2. Circle packing: 26 disjoint circles in the unit square, maximise the sum of radii.

AlphaEvolve problem (Novikov et al., arXiv 2506.13131: 2.63586275, previous best 2.634).
Later: FICO Xpress global solver 2.63591551 (FICO blog "FICO Xpress Optimization Surpasses
AlphaEvolve's Achievements", 2025); The Station (Chung & Du, arXiv 2511.06309) 2.63598.

Verifier: feasibility with absolute tolerance TOL. To stop tolerance abuse (inflating every
radius by TOL), the metric is the sum of radii minus N * (largest constraint violation):
shrinking every radius by that violation gives a strictly feasible packing, so the metric
is always attained by a truly feasible configuration.

Anchors: baseline = best equal-radius square grid (5 x 6 slots, r = 1/12, sum 13/6);
reference = best published value; m* = sqrt(N / pi), a proven upper bound (the circles are
disjoint inside the square, so pi * sum r^2 <= 1, and sum r <= sqrt(N * sum r^2) by
Cauchy-Schwarz). No tighter rigorous bound is known to us; with this m* scores above 0.5
grow slowly, which is intended (records here move the fifth decimal).
"""

from __future__ import annotations

import math

import numpy as np

from .base import Anchors, InvalidSubmission, Problem, ProblemInfo

N = 26
TOL = 1e-9
# The Station reports 2.63598 (5 d.p.); the best configuration we reproduced sums to
# 2.6359830849..., so the stored reference is that value rounded up at the 7th decimal:
# reproducing the published record scores <= 0.5 and only a real improvement counts as one.
REFERENCE = 2.6359831
UPPER_BOUND = math.sqrt(N / math.pi)


def violation(c: np.ndarray) -> float:
    """Largest constraint violation (<= 0 means strictly feasible) of an (N, 3) array."""
    x, y, r = c.T
    i, j = np.triu_indices(len(c), 1)
    overlap = r[i] + r[j] - np.hypot(x[i] - x[j], y[i] - y[j])
    wall = np.concatenate([r - x, x + r - 1, r - y, y + r - 1])
    return float(max(overlap.max(), wall.max()))


def objective(c: np.ndarray) -> float:
    return float(c[:, 2].sum() - len(c) * max(0.0, violation(c)))


def grid_packing(n: int = N) -> list[list[float]]:
    """Builder baseline: equal circles on the best a x b square grid with a*b >= n."""
    a = min(range(1, n + 1), key=lambda a: max(a, math.ceil(n / a)))
    b = math.ceil(n / a)
    m = max(a, b)
    r = 1 / (2 * m)
    cells = [((i + 0.5) / m, (j + 0.5) / m) for j in range(b) for i in range(a)]
    return [[x, y, r] for x, y in cells[:n]]


RECORD_MARGIN = 0.001  # decision: how large a record improvement earns 1.0


class CirclePacking26(Problem):
    info = ProblemInfo(
        id="M2",
        domain="math",
        title="Circle packing in a square (n = 26, max sum of radii)",
        kind="exact",
        metric="sum of radii (higher is better)",
        higher_is_better=True,
    )

    def statement(self) -> str:
        return f"""# M2 · Circle packing: 26 circles in the unit square

Place {N} pairwise disjoint circles inside the unit square [0, 1]^2 so that the **sum of
their radii is as large as possible**. Radii may differ.

**Submission** (JSON): `{{"circles": [[x, y, r], ...]}}` with exactly {N} entries; (x, y) is
the centre, r > 0 the radius.

**Constraints** (checked with absolute tolerance {TOL:g}):
r ≤ x ≤ 1 - r, r ≤ y ≤ 1 - r for every circle, and
sqrt((x_i - x_j)^2 + (y_i - y_j)^2) ≥ r_i + r_j for every pair.
A submission violating any constraint by more than {TOL:g} is rejected.

**Exact objective:** `sum(r) - {N} * max(0, v)` where v is the largest constraint violation
(so the tolerance cannot be used to inflate radii; a strictly feasible packing scores
exactly its sum of radii). Use full float precision in your JSON.

An equal-radius 5 x 6 grid gives 13/6 ≈ 2.1667. The best published value is about
2.63598; the trivial area bound is sqrt({N}/π) ≈ {UPPER_BOUND:.4f}. This problem has no
hidden data: `lab eval` returns the exact objective.
"""

    def anchors(self) -> Anchors:
        return Anchors(
            baseline=objective(np.asarray(grid_packing())),
            reference=REFERENCE,
            optimum=UPPER_BOUND,
            # 1.0 = beating the record by RECORD_MARGIN (a proven-bound emin would make records ~flat)
            emin=UPPER_BOUND - (REFERENCE + RECORD_MARGIN),
            ref_kind="P",
            ref_source="The Station, arXiv 2511.06309 (2.63598; FICO Xpress 2.63591551, "
            "AlphaEvolve 2.63586275); m* = sqrt(26/pi) area bound",
        )

    def verify(self, submission) -> float:
        c = submission.get("circles") if isinstance(submission, dict) else None
        if not isinstance(c, list) or len(c) != N:
            raise InvalidSubmission(f"'circles' must be a list of exactly {N} [x, y, r] triples")
        if any(not isinstance(row, list) or len(row) != 3
               or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in row) for row in c):
            raise InvalidSubmission("each circle must be a list of 3 numbers [x, y, r]")
        c = np.asarray(c, dtype=float)
        if not np.all(np.isfinite(c)):
            raise InvalidSubmission("circle coordinates and radii must be finite")
        if np.any(c[:, 2] <= 0):
            raise InvalidSubmission("every radius must be > 0")
        v = violation(c)
        if v > TOL:
            raise InvalidSubmission(f"infeasible packing: largest violation {v:.3g} > {TOL:g}")
        return objective(c)

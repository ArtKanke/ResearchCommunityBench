"""M1. First autocorrelation inequality (AlphaEvolve problem list, arXiv 2511.02864).

For non-negative f supported on [-1/4, 1/4], minimise C = max_t (f*f)(t) / (int f)^2.
Agents submit a non-negative step function with n equal-width pieces. For equal widths
w = 1/(2n), f*f is piecewise linear with knots on the grid, so the maximum is attained
at a knot: max (f*f) = w * max(conv(h, h)), and C = 2n * max(conv(h, h)) / (sum h)^2.
"""

from __future__ import annotations

import numpy as np

from .base import Anchors, InvalidSubmission, Problem, ProblemInfo

MAX_PIECES = 20_000


def objective(h: np.ndarray) -> float:
    n = len(h)
    h = h / np.max(h)  # scale-invariant; normalising blocks underflow tricks (cf. AlphaEvolve notebook)
    return float(2 * n * np.max(np.convolve(h, h)) / np.sum(h) ** 2)


class AutocorrelationM1(Problem):
    info = ProblemInfo(
        id="M1",
        domain="math",
        title="First autocorrelation inequality",
        kind="exact",
        metric="C (lower is better)",
        higher_is_better=False,
    )

    def statement(self) -> str:
        return f"""# M1 · First autocorrelation inequality

For a non-negative function f supported on [-1/4, 1/4], let

    C(f) = max_t (f * f)(t) / (∫ f)^2

where f * f is the autoconvolution. Find f with C(f) as small as possible.

**Submission** (JSON): `{{"heights": [h_1, ..., h_n]}}`, the heights of a step function with
n equal-width pieces covering [-1/4, 1/4]. All h_i ≥ 0, at least one > 0, 1 ≤ n ≤ {MAX_PIECES}.

**Exact objective** (what the verifier computes): with w = 1/(2n),
`C = 2n · max(np.convolve(h, h)) / (sum h)^2`.

The constant function gives C = 2. The best known constructions are near 1.50; the best
known lower bound is about 1.28. This problem has no hidden data: `lab eval` returns the
exact objective.
"""

    def anchors(self) -> Anchors:
        return Anchors(
            baseline=np.pi / 2,
            reference=1.503163554681561,
            optimum=1.2802,
            emin=1.5000 - 1.2802,  # 1.0 = C <= 1.5000, ~0.003 below the record (decision)
            ref_kind="P",
            ref_source="AlphaEvolve, arXiv 2511.02864 (prev. 1.50992); lower bound 1.2802",
        )

    def verify(self, submission) -> float:
        h = submission.get("heights") if isinstance(submission, dict) else None
        if not isinstance(h, list) or not 1 <= len(h) <= MAX_PIECES:
            raise InvalidSubmission(f"'heights' must be a list of 1..{MAX_PIECES} numbers")
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)) or np.any(h < 0) or h.sum() <= 0:
            raise InvalidSubmission("heights must be finite, non-negative and not all zero")
        return objective(h)

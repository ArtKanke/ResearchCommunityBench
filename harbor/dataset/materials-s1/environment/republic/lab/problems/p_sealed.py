"""P1/P2. Sealed dynamical systems (MysteryMechanism-style law discovery, continuous score).

Each seed draws a hidden ODE. Agents spend a shared budget of noisy simulations, then
predict trajectories from public test initial conditions, some outside the region the
oracle allows (extrapolation rewards finding the true law, not curve fitting).

Metric: NMSE of the predicted (measured) variables over the test set, normalised by the
variance of the true test values. Baseline = linear least-squares model fitted from the
same oracle budget (builder, per instance); reference = a fixed target of 1% NMSE.
Only the first state variable is measured (P1: position; P2: prey); the other is latent.
P1 draws 3 of 7 hidden terms. Score 1.0 at NMSE 1e-7 (near the integration floor).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from .base import Anchors, InvalidSubmission, OracleSpec, Problem, ProblemInfo

DT = 0.1
T_MAX = 10.0
NOISE = 0.01
P1_NOISE, P1_BUDGET = 0.05, 16  # tuned after Codex solved both systems perfectly in 15 min (Nov 14 pilot)
P2_BUDGET = 16  # was 32; P2 still scored ~1.0 in every arm of the Nov 14 trio (untested at 16)
REF_NMSE = 0.01  # score 0.5 = 1% normalised error
TEST_TIMES = [round(0.5 * k, 1) for k in range(1, 11)]  # 0.5 .. 5.0
N_TEST = 60
BUDGET = 64


def _simulate(rhs, y0, t_eval):
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10, method="LSODA")
    y = sol.y.T
    if y.shape[0] < len(t_eval):  # integration failed (blow-up): pad with last state
        pad = np.repeat(y[-1:] if len(y) else np.asarray([y0]), len(t_eval) - y.shape[0], axis=0)
        y = np.vstack([y, pad])
    return y


def _oscillator(rng):
    k1 = rng.uniform(0.5, 2.0)
    g = rng.uniform(0.05, 0.3)
    pool = {
        "cubic": lambda x, v, c: -c * x**3,
        "pendulum": lambda x, v, c: -c * np.sin(2 * x),
        "drag": lambda x, v, c: -c * v * np.abs(v),
        "vdp": lambda x, v, c: -c * (x**2 - 1) * v,
        "coulomb": lambda x, v, c: -c * np.tanh(3 * v),
        "well": lambda x, v, c: -2 * c * x * np.exp(-x**2),
        "quadratic": lambda x, v, c: -c * x * np.abs(x),
    }
    names = list(rng.choice(list(pool), size=3, replace=False))
    coefs = {n: rng.uniform(0.2, 1.0) for n in names}

    def rhs(t, y):
        x, v = y
        a = -k1 * x - g * v + sum(pool[n](x, v, coefs[n]) for n in names)
        return [v, a]

    box = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    test_box = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    return rhs, ["x", "v"], box, test_box, {"k1": k1, "g": g, "terms": coefs}


def _predator_prey(rng):
    r, K = rng.uniform(0.8, 1.5), rng.uniform(2.0, 4.0)
    a, b, m = rng.uniform(0.6, 1.2), rng.uniform(0.4, 0.9), rng.uniform(0.2, 0.5)
    h = rng.uniform(0.3, 1.0)
    q = rng.uniform(0.0, 0.6) if rng.random() < 0.5 else 0.0  # Beddington-DeAngelis interference

    def rhs(t, y):
        x, z = max(y[0], 0.0), max(y[1], 0.0)
        f = x / (1 + h * x + q * z)
        return [r * x * (1 - x / K) - a * f * z, b * f * z - m * z]

    box = np.array([[0.1, 3.0], [0.1, 3.0]])
    test_box = np.array([[0.1, 4.5], [0.1, 4.5]])
    return rhs, ["prey", "predator"], box, test_box, {"r": r, "K": K, "a": a, "b": b, "m": m, "h": h, "q": q}


GENERATORS = {"P1": _oscillator, "P2": _predator_prey}


def _poly_features(Y, degree):
    cols = [np.ones(len(Y))]
    d = Y.shape[1]
    for i in range(d):
        cols.append(Y[:, i])
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(Y[:, i] * Y[:, j])
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(Y[:, i] * Y[:, j] * Y[:, k])
    return np.column_stack(cols)


def _fit_and_predict(trajs, test_y0, degree, t_eval):
    """Least squares on central differences of noisy trajectories, then integrate."""
    X, dY = [], []
    for y in trajs:
        X.append(y[1:-1])
        dY.append((y[2:] - y[:-2]) / (2 * DT))
    X, dY = np.vstack(X), np.vstack(dY)
    F = _poly_features(X, degree)
    W = np.linalg.lstsq(F, dY, rcond=None)[0]
    if degree >= 2:  # sequentially thresholded least squares (SINDy)
        for _ in range(5):
            small = np.abs(W) < 0.02
            W[small] = 0
            for j in range(W.shape[1]):
                keep = ~small[:, j]
                if keep.any():
                    W[keep, j] = np.linalg.lstsq(F[:, keep], dY[:, j], rcond=None)[0]

    def rhs(t, y):
        out = _poly_features(np.asarray([y]), degree) @ W
        return np.clip(out[0], -1e3, 1e3)

    return np.array([_simulate(rhs, y0, t_eval) for y0 in test_y0])


class SealedSystem(Problem):
    def __init__(self, seed: int = 0, pid: str = "P1"):
        super().__init__(seed)
        self.pid = pid
        rng = np.random.default_rng(10_000 * (1 if pid == "P1" else 2) + seed)
        self.rhs, self.names, self.box, self.test_box, self._secret = GENERATORS[pid](rng)
        self.obs = [0]  # only the first variable is measured (P1 position, P2 prey); the other is latent
        self.noise = P1_NOISE if pid == "P1" else 0.03
        self.budget = P1_BUDGET if pid == "P1" else P2_BUDGET
        self.test_y0 = rng.uniform(self.test_box[:, 0], self.test_box[:, 1], size=(N_TEST, len(self.names)))
        self.t_eval = np.array([0.0] + TEST_TIMES)
        self._noise_rng = np.random.default_rng(seed + 7)
        title = "Sealed oscillator, position only" if pid == "P1" else "Sealed predator-prey system"
        self.info = ProblemInfo(
            id=pid,
            domain="physics",
            title=title,
            kind="generalization",
            metric="NMSE on hidden trajectories (lower is better)",
            higher_is_better=False,
            dev_budget=0,
            oracle=OracleSpec(budget=self.budget, description="one noisy simulation per query"),
        )

    @property
    def observed(self):
        return [self.names[i] for i in self.obs]

    @property
    def truth(self):
        return _truth(self.pid, self.seed)

    def statement(self) -> str:
        n = ", ".join(self.names)
        lo = ", ".join(f"{a:g}..{b:g}" for a, b in self.box)
        partial = (f" Only **{', '.join(self.observed)}** is measured; you set every initial state "
                   f"coordinate but never observe the others.") if len(self.obs) < len(self.names) else ""
        return f"""# {self.pid} · {self.info.title}

A hidden autonomous ODE system with state ({n}). Discover how it behaves and predict it.{partial}

**Oracle** (`lab oracle {self.pid} '{{"y0": [..], "t_end": 10}}'`): runs one experiment from
initial state y0 (each coordinate within {lo}) and returns the measured variables
({", ".join(self.observed)}) every {DT} time units up to t_end ≤ {T_MAX:g}, with Gaussian
measurement noise (sd {self.noise}). The whole institute shares **{self.budget} experiments**.

**Task:** `test_inputs.json` lists {N_TEST} initial states, some outside the oracle's range.
Predict the measured variables at times {TEST_TIMES}.

**Submission** (JSON): `{{"predictions": P}}` where P[i][k] is a list of {len(self.obs)} number(s)
({", ".join(self.observed)}) for test initial state i at time TEST_TIMES[k].

**Metric:** mean squared error normalised by the variance of the true test values (NMSE);
0 is perfect. Baseline (score 0): a linear model fitted from the same budget. Score 0.5 at
1% NMSE. The true law is not limited to polynomials.
"""

    def public_files(self):
        return {
            "test_inputs.json": json.dumps(
                {"state": self.names, "measured": self.observed, "times": TEST_TIMES, "y0": self.test_y0.round(6).tolist()}
            ).encode()
        }

    def _measure(self, y0, t, rng):
        y = _simulate(self.rhs, y0, t)[:, self.obs]
        return y + rng.normal(0, self.noise, size=y.shape)

    def oracle(self, query):
        y0 = np.asarray(query.get("y0", []), dtype=float)
        t_end = float(query.get("t_end", T_MAX))
        if y0.shape != (len(self.names),) or not np.all(np.isfinite(y0)):
            raise InvalidSubmission(f"y0 must be a list of {len(self.names)} numbers")
        if np.any(y0 < self.box[:, 0]) or np.any(y0 > self.box[:, 1]):
            raise InvalidSubmission(f"y0 outside the oracle range {self.box.tolist()}")
        if not 0 < t_end <= T_MAX:
            raise InvalidSubmission(f"t_end must be in (0, {T_MAX}]")
        t = np.round(np.arange(0, t_end + 1e-9, DT), 6)
        return {"t": t.tolist(), "measured": self.observed, "y": self._measure(y0, t, self._noise_rng).round(5).tolist()}

    def _nmse(self, pred):
        true = self.truth[:, 1:, :][:, :, self.obs]
        var = true.reshape(-1, true.shape[-1]).var(axis=0)
        err = np.clip(pred - true, -1e6, 1e6) ** 2
        return float(np.mean(err / var))

    def verify(self, submission) -> float:
        p = submission.get("predictions") if isinstance(submission, dict) else None
        try:
            pred = np.asarray(p, dtype=float)
        except (TypeError, ValueError):
            raise InvalidSubmission("predictions must be numeric")
        want = (N_TEST, len(TEST_TIMES), len(self.obs))
        if pred.shape != want or not np.all(np.isfinite(pred)):
            raise InvalidSubmission(f"predictions must have shape {list(want)} and be finite")
        return self._nmse(pred)

    def anchors(self) -> Anchors:
        base, ref, how = _builder_anchors(self.pid, self.seed)
        return Anchors(baseline=base, reference=ref, optimum=0.0, emin=1e-7, ref_kind="B",
                       ref_source=f"builder at the same budget: linear LSQ (0) / {how} (0.5)")


@lru_cache(maxsize=32)
def _truth(pid, seed):
    p = SealedSystem(seed, pid)
    return np.array([_simulate(p.rhs, y0, p.t_eval) for y0 in p.test_y0])


@lru_cache(maxsize=32)
def _builder_anchors(pid, seed):
    from scipy.signal import savgol_filter

    p = SealedSystem(seed, pid)
    rng = np.random.default_rng(seed + 99)
    t = np.round(np.arange(0, T_MAX + 1e-9, DT), 6)
    trajs = []
    for _ in range(p.budget):
        y0 = rng.uniform(p.box[:, 0], p.box[:, 1])
        m = p._measure(y0, t, rng)
        if p.pid == "P1":  # rebuild velocity from smoothed position
            x = savgol_filter(m[:, 0], 9, 3)
            m = np.column_stack([x, savgol_filter(m[:, 0], 9, 3, deriv=1, delta=DT)])
        trajs.append(m)
    if p.pid == "P1":
        lin = _fit_and_predict(trajs, p.test_y0, 1, p.t_eval)[:, 1:, :][:, :, p.obs]
    else:  # P2: a linear model of the measured prey alone (the predator is latent)
        lin = _fit_and_predict(trajs, p.test_y0[:, p.obs], 1, p.t_eval)[:, 1:, :]
    base = p._nmse(lin)
    # Builder polynomial fits proved too erratic across seeds to anchor 0.5 (sometimes no better
    # than linear), so the reference is a fixed target: 1% normalised error (decision; see SUMMARY).
    ref = min(REF_NMSE, base / 5)
    return base, ref, f"fixed target NMSE {ref:.3g}"

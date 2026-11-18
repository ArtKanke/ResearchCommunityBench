"""X2. Global minimum of a binary Lennard-Jones cluster A19B19 with seed-specific parameters.

Binary LJ clusters (Doye & Meloni, Chem. Phys. Lett. 406, 2005; Cambridge Cluster Database)
are tabulated only for additive sigma_AB = (1 + sigma_BB)/2 and eps_AA = eps_AB = eps_BB = 1.
Here every seed draws non-additive sigma_AB and unequal eps_AB, eps_BB, so no database value
applies and the instance must be solved from scratch.

E = sum_{i<j} 4 eps_ij [(sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6], no cutoff, sigma_AA =
eps_AA = 1. The verifier evaluates E on the submitted coordinates as given (no server-side
relaxation), rejecting non-finite input and pairs closer than 0.5 sigma_ij.

Anchors (builder, computed per seed and cached): baseline = best of 5 random starts +
L-BFGS; reference = basin hopping (random displacements, A/B swaps, surface jumps) with a
fixed step count; m* = reference - 5% |reference|, a deliberately loose target below any
plausible global minimum (for the homo-LJ38 case the global minimum is ~1% below what a
modest basin hopping run typically finds).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .base import Anchors, InvalidSubmission, Problem, ProblemInfo

N_A, N_B = 19, 19
N = N_A + N_B
MIN_DIST = 0.5  # in units of sigma_ij
BH_STEPS = 2500
BH_TEMP = 0.8
_IU = np.triu_indices(N, 1)


def draw_params(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(20_000 + seed)
    s_bb = rng.uniform(0.80, 1.25)
    e_bb = rng.uniform(0.5, 1.5)
    s_ab = 0.5 * (1 + s_bb) * rng.uniform(0.93, 1.07)  # non-additive
    e_ab = np.sqrt(e_bb) * rng.uniform(0.8, 1.5)  # non-Berthelot
    return {"sigma_AB": round(float(s_ab), 6), "eps_AB": round(float(e_ab), 6),
            "sigma_BB": round(float(s_bb), 6), "eps_BB": round(float(e_bb), 6)}


def pair_params(types: np.ndarray, p: dict[str, float]):
    """types: int array (0 = A, 1 = B). Returns per-pair sigma, eps over the upper triangle."""
    S = np.array([[1.0, p["sigma_AB"]], [p["sigma_AB"], p["sigma_BB"]]])
    E = np.array([[1.0, p["eps_AB"]], [p["eps_AB"], p["eps_BB"]]])
    a, b = types[_IU[0]], types[_IU[1]]
    return S[a, b], E[a, b]


def energy_grad(xf: np.ndarray, s: np.ndarray, e: np.ndarray):
    X = xf.reshape(N, 3)
    d = X[_IU[0]] - X[_IU[1]]
    r2 = (d * d).sum(1)
    q = (s * s / r2) ** 3
    gp = (2 * 4 * e * (3 * q - 6 * q * q) / r2)[:, None] * d
    G = np.zeros((N, 3))
    np.add.at(G, _IU[0], gp)
    np.add.at(G, _IU[1], -gp)
    return float((4 * e * (q * q - q)).sum()), G.ravel()


def energy(X: np.ndarray, types: np.ndarray, p: dict[str, float]) -> float:
    return energy_grad(X.ravel(), *pair_params(types, p))[0]


def relax(X: np.ndarray, types: np.ndarray, p: dict[str, float]):
    s, e = pair_params(types, p)
    r = minimize(energy_grad, X.ravel(), args=(s, e), jac=True, method="L-BFGS-B",
                 options={"maxiter": 3000, "gtol": 1e-7})
    return float(r.fun), r.x.reshape(N, 3)


def random_start(rng: np.random.Generator) -> np.ndarray:
    R = 0.9 * (N / (4 / 3 * np.pi * 0.8)) ** (1 / 3)
    while True:
        X = rng.uniform(-R, R, (3 * N, 3))
        X = X[(X**2).sum(1) < R * R]
        if len(X) >= N:
            return X[:N]


def default_types() -> np.ndarray:
    return np.array([0] * N_A + [1] * N_B)


def random_relaxed(p, n_starts: int, rng) -> tuple[float, np.ndarray, np.ndarray]:
    t = default_types()
    runs = [(*relax(random_start(rng), t, p), t) for _ in range(n_starts)]
    return min(runs, key=lambda r: r[0])


def basin_hop(p, steps: int, rng, start=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Monte Carlo on local minima: displacements, A/B swaps, surface jumps (Metropolis)."""
    Ec, Xc, tc = start if start is not None else random_relaxed(p, 1, rng)
    best = (Ec, Xc, tc)
    for _ in range(steps):
        Xn, tn = Xc.copy(), tc.copy()
        u = rng.random()
        if u < 0.3:
            i, j = rng.choice(np.flatnonzero(tn == 0)), rng.choice(np.flatnonzero(tn == 1))
            tn[i], tn[j] = 1, 0
        elif u < 0.5:
            c = Xn.mean(0)
            v = rng.normal(size=3)
            Xn[rng.integers(N)] = c + v / np.linalg.norm(v) * np.linalg.norm(Xn - c, axis=1).max()
        else:
            Xn += rng.uniform(-0.4, 0.4, Xn.shape)
        En, Xr = relax(Xn, tn, p)
        if En < Ec or rng.random() < np.exp(-(En - Ec) / BH_TEMP):
            Ec, Xc, tc = En, Xr, tn
            if En < best[0] - 1e-9:
                best = (En, Xr, tn)
    return best


@lru_cache(maxsize=32)
def builder(seed: int):
    """(baseline, reference) as (energy, positions, types) tuples for one seed."""
    p = draw_params(seed)
    base = random_relaxed(p, 5, np.random.default_rng(seed + 1))
    ref = basin_hop(p, BH_STEPS, np.random.default_rng(seed + 2))
    return base, ref


def submission_of(X: np.ndarray, types: np.ndarray) -> dict:
    return {"positions": X.tolist(), "types": ["AB"[t] for t in types]}


class BinaryLJCluster(Problem):
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.params = draw_params(seed)
        self.info = ProblemInfo(
            id="X2",
            domain="materials",
            title=f"Binary Lennard-Jones cluster A{N_A}B{N_B} (non-standard parameters)",
            kind="exact",
            metric="total LJ energy (lower is better)",
            higher_is_better=False,
        )

    def statement(self) -> str:
        p = self.params
        return f"""# X2 · Binary Lennard-Jones cluster A{N_A}B{N_B}

Find the lowest-energy configuration of {N} atoms, exactly {N_A} of type A and {N_B} of
type B, interacting through pairwise Lennard-Jones potentials (no cutoff, no periodic box):

    E = sum over pairs i<j of 4 eps_ij [ (sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6 ]

| pair | sigma | eps |
|------|-------|-----|
| AA | 1 | 1 |
| AB | {p['sigma_AB']} | {p['eps_AB']} |
| BB | {p['sigma_BB']} | {p['eps_BB']} |

The parameters are non-additive and instance-specific: published cluster databases do not
apply. You choose both the geometry and which atoms are A or B.

**Submission** (JSON): `{{"positions": [[x, y, z], ...], "types": ["A" | "B", ...]}}` with
{N} positions and {N} matching types ({N_A} "A", {N_B} "B").

**Exact objective:** E evaluated on the submitted coordinates as given (no relaxation on
the server, so relax locally and submit full float precision). Submissions with non-finite
coordinates or any pair closer than {MIN_DIST} sigma_ij are rejected. Lower is better.
A few random starts with local relaxation are the baseline; a modest basin-hopping run is
the reference. This problem has no hidden data: `lab eval` returns the exact objective.
"""

    def anchors(self) -> Anchors:
        (eb, *_), (er, *_) = builder(self.seed)
        return Anchors(
            baseline=eb,
            reference=er,
            optimum=er - 0.05 * abs(er),
            ref_kind="B",
            ref_source=f"builder: best of 5 random relaxations (0) / {BH_STEPS}-step basin "
            "hopping (0.5); m* = reference - 5% |reference|",
        )

    def verify(self, submission) -> float:
        if not isinstance(submission, dict):
            raise InvalidSubmission("submission must be a JSON object")
        pos, types = submission.get("positions"), submission.get("types")
        if not isinstance(pos, list) or len(pos) != N:
            raise InvalidSubmission(f"'positions' must be a list of {N} [x, y, z] triples")
        if not isinstance(types, list) or len(types) != N or any(t not in ("A", "B") for t in types):
            raise InvalidSubmission(f"'types' must be a list of {N} strings 'A' or 'B'")
        t = np.array([0 if s == "A" else 1 for s in types])
        if (t == 0).sum() != N_A:
            raise InvalidSubmission(f"composition must be {N_A} A + {N_B} B")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               for row in pos for v in (row if isinstance(row, list) else [None])):
            raise InvalidSubmission("positions must be lists of numbers")
        X = np.asarray(pos, dtype=float) if all(len(r) == 3 for r in pos) else None
        if X is None or X.shape != (N, 3) or not np.all(np.isfinite(X)):
            raise InvalidSubmission(f"positions must have shape [{N}, 3] and be finite")
        s, e = pair_params(t, self.params)
        r = np.linalg.norm(X[_IU[0]] - X[_IU[1]], axis=1)
        if np.any(r < MIN_DIST * s):
            raise InvalidSubmission(f"atoms closer than {MIN_DIST} sigma_ij")
        E = energy_grad(X.ravel(), s, e)[0]
        if not np.isfinite(E):
            raise InvalidSubmission("energy is not finite")
        return E

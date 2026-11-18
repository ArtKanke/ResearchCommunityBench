"""B1. Ehrlich holo function design (Stanton et al. 2024, "Closed-Form Test Functions for
Biophysical Sequence Design", arXiv 2407.00236; reference code prescient-design/holo-bench,
Apache-2.0, not used: its pip package pins numpy<2 and pulls torch/botorch).

Implemented from the paper's definition. A sequence x in V^L is feasible iff every adjacent
pair is an allowed transition of a hidden sparse first-order transition matrix (the DMP;
every state keeps its self-transition and a hidden Hamiltonian cycle keeps it ergodic).
c hidden motifs a^(i) in V^k with hidden spacings s^(i) (s_1 = 0 < s_2 < ... < s_k < L/c):

    h_q(x, a, s) = max_l floor(q * #{j : x[l + s_j] = a_j} / k) / q
    f(x) = prod_i h_q(x, a^(i), s^(i))  if x feasible, else 0

Motif elements are distinct across motifs and the instance is resampled until an optimal
sequence provably exists (constructed explicitly), so max f = 1 exactly.
Anchors per seed: baseline = best of BUDGET random feasible walks; reference = mean over
3 runs of a simple genetic algorithm (truncation selection, one-point crossover, point
mutation) at the same budget.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from .base import Anchors, InvalidSubmission, OracleSpec, Problem, ProblemInfo

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
V = len(ALPHABET)
K = 4
Q = K
BUDGET = 2000
MAX_BATCH = 50
N_BAN = 6  # banned next-states per state


def _dist(feas):
    """All-pairs shortest path lengths on the transition graph (BFS)."""
    d = np.full((V, V), 10**6, dtype=int)
    for a in range(V):
        d[a, a] = 0
        dq = deque([a])
        while dq:
            u = dq.popleft()
            for w in np.flatnonzero(feas[u]):
                if d[a, w] > d[a, u] + 1:
                    d[a, w] = d[a, u] + 1
                    dq.append(w)
    return d


def _path(feas, a, b, n):
    """Feasible walk of exactly n steps from a to b (inclusive endpoints), or None."""
    prev = {a: None}
    dq = deque([a])
    while dq:
        u = dq.popleft()
        if u == b:
            break
        for w in np.flatnonzero(feas[u]):
            if w not in prev:
                prev[w] = u
                dq.append(w)
    if b not in prev:
        return None
    p = [b]
    while prev[p[-1]] is not None:
        p.append(prev[p[-1]])
    p = p[::-1]
    if len(p) - 1 > n:
        return None
    return [a] * (n - len(p) + 1) + p  # pad with self-transitions


class Ehrlich:
    def __init__(self, rng: np.random.Generator, L: int, c: int):
        self.L, self.c = L, c
        while True:
            feas = np.ones((V, V), dtype=bool)
            cyc = rng.permutation(V)
            succ = np.empty(V, dtype=int)
            succ[cyc] = np.roll(cyc, -1)
            for a in range(V):
                cand = [b for b in range(V) if b != a and b != succ[a]]
                feas[a, rng.choice(cand, size=N_BAN, replace=False)] = False
            elems = rng.choice(V, size=c * K, replace=False).reshape(c, K)
            seg = L // c
            spac = np.array([np.r_[0, np.sort(rng.choice(np.arange(1, seg), size=K - 1, replace=False))]
                             for _ in range(c)])
            self.feas, self.motifs, self.spacings = feas, elems, spac
            w = self._construct()
            if w is not None:
                self.witness = w
                break

    def _construct(self):
        L, seg = self.L, self.L // self.c
        fixed = {}
        for i in range(self.c):
            for j in range(K):
                fixed[i * seg + int(self.spacings[i, j])] = int(self.motifs[i, j])
        pos = sorted(fixed)
        x = [fixed[pos[0]]] * (pos[0] + 1)
        for p, p2 in zip(pos, pos[1:]):
            walk = _path(self.feas, fixed[p], fixed[p2], p2 - p)
            if walk is None:
                return None
            x += walk[1:]
        x += [x[-1]] * (L - len(x))
        x = np.array(x)
        return x if self.f(x[None])[0] == 1.0 else None

    def feasible(self, X):
        return self.feas[X[:, :-1], X[:, 1:]].all(axis=1)

    def f(self, X):
        X = np.atleast_2d(X)
        out = np.ones(len(X))
        for a, s in zip(self.motifs, self.spacings):
            span = self.L - s[-1]
            m = sum((X[:, s[j]:s[j] + span] == a[j]) for j in range(K)).max(axis=1)
            out *= np.floor(Q * m / K) / Q
        return np.where(self.feasible(X), out, 0.0)

    def random_feasible(self, rng, n):
        X = np.empty((n, self.L), dtype=int)
        X[:, 0] = rng.integers(V, size=n)
        nxt = [np.flatnonzero(r) for r in self.feas]
        for t in range(1, self.L):
            X[:, t] = [rng.choice(nxt[a]) for a in X[:, t - 1]]
        return X


SHAPES = [(32, 3), (32, 4), (48, 3), (48, 4), (64, 4)]  # (L, c); GA reliably beats random here


def _instance(seed):
    rng = np.random.default_rng(31_000 + seed)
    L, c = SHAPES[rng.integers(len(SHAPES))]
    return Ehrlich(rng, L, c)


def _ga(e: Ehrlich, rng, budget=BUDGET, pop=MAX_BATCH, elite=20):
    """Truncation selection, one-point crossover, uniform point mutation at rate 1/L."""
    P = e.random_feasible(rng, pop)
    F = e.f(P)
    used = pop
    while used + pop <= budget:
        top = np.argsort(-F, kind="stable")[:elite]
        pa, pb = P[rng.choice(top, pop)], P[rng.choice(top, pop)]
        child = np.where(np.arange(e.L) < rng.integers(1, e.L, size=(pop, 1)), pa, pb)
        mut = rng.random(child.shape) < 1.0 / e.L
        child[mut] = rng.integers(V, size=mut.sum())
        Fc = e.f(child)
        used += pop
        P, F = np.vstack([P[top], child]), np.r_[F[top], Fc]
    return float(F.max())


@lru_cache(maxsize=32)
def _builder_anchors(seed):
    e = _instance(seed)
    rng = np.random.default_rng(seed + 4242)
    base = float(e.f(e.random_feasible(rng, BUDGET)).max())
    ref = float(np.mean([_ga(e, np.random.default_rng(seed * 10 + r)) for r in range(10)]))  # 10 GA runs: a 3-run mean had SE ~0.07
    return base, ref


def _decode(s):
    if not isinstance(s, str):
        raise InvalidSubmission("sequences must be strings")
    try:
        return [ALPHABET.index(ch) for ch in s]
    except ValueError:
        raise InvalidSubmission(f"sequences may only use the letters {ALPHABET}")


class EhrlichDesign(Problem):
    info = ProblemInfo(
        id="B1",
        domain="biology",
        title="Ehrlich holo sequence design",
        kind="exact",
        metric="f(x) of the submitted sequence (higher is better, max 1)",
        higher_is_better=True,
        oracle=OracleSpec(budget=BUDGET, description=f"f values for up to {MAX_BATCH} sequences per "
                                                      "query; each sequence costs 1 unit"),
    )

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._e = _instance(seed)
        self.L, self.c = self._e.L, self._e.c

    def _parse(self, seqs):
        X = np.array([_decode(s) for s in seqs])
        if X.ndim != 2 or X.shape[1] != self.L:
            raise InvalidSubmission(f"every sequence must have length {self.L}")
        return X

    def statement(self) -> str:
        return f"""# B1 · Ehrlich holo sequence design

Design a sequence that maximises a hidden **Ehrlich function** (Stanton et al. 2024,
arXiv 2407.00236), a closed-form model of biophysical sequence design with feasibility
constraints and epistasis.

- Alphabet: the {V} letters `{ALPHABET}`; sequence length **L = {self.L}**.
- **Feasibility:** a hidden first-order transition rule bans some ordered pairs of
  adjacent letters (each letter forbids {N_BAN} of the {V} possible successors; repeating a
  letter is always allowed). Any sequence containing a banned adjacent pair has f = 0.
- **Objective:** there are c = {self.c} hidden motifs, each of k = {K} distinct letters with a
  hidden spacing vector s (s_1 = 0 < s_2 < s_3 < s_4 < L/c = {self.L // self.c}). For one motif,
  h = max over offsets l of (number of j with x[l + s_j] = a_j) / {K}, quantised to multiples
  of 1/q, q = {Q}. For feasible x, f(x) = product of the c motif scores, so f takes values in
  products of {{0, 1/4, 1/2, 3/4, 1}}. No two motifs share a letter.
- The maximum f = 1 is attainable (all motifs fully present, sequence feasible).

**Oracle** (`lab oracle B1 '{{"sequences": ["ACDE...", ...]}}'`): returns
`{{"values": [f(x) ...]}}` for up to {MAX_BATCH} sequences per call. Each sequence costs one unit
of the institute's shared budget of **{BUDGET} evaluations** (infeasible ones too).

**Submission** (JSON): `{{"sequence": "..."}}`, one string of length {self.L}. The score is the
exact f of that sequence, so `lab eval` returns it. Baseline: best of {BUDGET} random feasible
sequences; reference: a simple genetic algorithm at the same budget.
"""

    def anchors(self) -> Anchors:
        base, ref = _builder_anchors(self.seed)
        return Anchors(baseline=base, reference=ref, optimum=1.0, emin=1e-3, ref_kind="B",
                       ref_source=f"builder: best of {BUDGET} random feasible (0) / GA at {BUDGET} evals, "
                                  "mean of 3 runs (0.5)")

    def oracle(self, query):
        seqs = query.get("sequences") if isinstance(query, dict) else None
        if not isinstance(seqs, list) or not 1 <= len(seqs) <= MAX_BATCH:
            raise InvalidSubmission(f"'sequences' must be a list of 1..{MAX_BATCH} strings")
        X = self._parse(seqs)
        return {"_cost": len(seqs), "values": self._e.f(X).tolist()}

    def verify(self, submission) -> float:
        s = submission.get("sequence") if isinstance(submission, dict) else None
        if not isinstance(s, str):
            raise InvalidSubmission("submission must be {'sequence': '<string>'}")
        return float(self._e.f(self._parse([s]))[0])

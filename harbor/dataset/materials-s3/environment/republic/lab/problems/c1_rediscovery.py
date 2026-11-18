"""C1. Sealed-target rediscovery (PMO-style sample-efficient molecular optimisation).

Each seed seals one drug-like molecule drawn from a held-back slice of ZINC-250k that is
never shipped to agents. The only way to learn anything about it is the oracle, which
returns Tanimoto similarity (Morgan radius 2, 2048 bits) of queried molecules to the
sealed target. The institute has 2000 molecule evaluations for the whole run, so the
problem is about sample efficiency, not about how long you can run a generative model.

Metric: PMO "AUC top-10" over the oracle query sequence (Gao et al., *Sample Efficiency
Matters: A Benchmark for Practical Molecular Optimization*, arXiv:2206.12411; reference
implementation `top_auc` in github.com/wenhao-gao/mol_opt `main/optimizer.py`). We follow
mol_opt's semantics -- running mean of the top-10 property values over the prefix of the
query sequence, integrated along the oracle-call axis and divided by the full budget, with
the final value padded out to the budget (mol_opt's `finish=True`) -- with three deliberate
deviations, all documented in the statement:

1. mol_opt samples the curve every `freq_log=100` calls and integrates trapezoidally from
   (0, 0); we sample every call and use right-endpoint rectangles. mol_opt's grid caps the
   attainable AUC at 1 - freq_log/(2*max_oracle_calls) = 0.995; ours caps it at
   1 - (TOP_K - 1)/(2*BUDGET) = 0.99775, the cost of the ten queries it takes to put ten
   molecules in the top ten. So `optimum = 1.0` is 0.00225 above anything attainable, which
   is immaterial next to emin = 0.3.
2. mol_opt's x-axis counts *unique valid* molecules, so duplicates and invalid SMILES are
   free. Here every molecule in a query costs one budget unit, valid or not, so wasted
   queries are paid for. The running top-10 is taken over distinct fingerprints (stereoisomers count once),
   so re-querying one good molecule cannot fill the top-10 with itself.
3. The running top-10 sum is always divided by 10, where mol_opt averages over however
   many molecules exist so far. mol_opt's grid hides the difference (its first sample is
   at 100 calls); sampling every call does not, and averaging over what exists would make
   the metric non-monotone and hand a run that queried one lucky molecule and stopped the
   same score as similarity 1.0 for 2000 calls. Dividing by 10 makes the running value
   non-decreasing, so an extra query is never worse than not making it.
4. The budget is 2000, not 10000: these runs are hours long, not days.

Anchors (builder-computed per seed, cached in data/C1/anchors_seed<k>.json):
baseline = random sampling from the public pool at the same budget, reference = a compact
graph GA (fragment crossover on acyclic single bonds + fragment-library mutation, in the
spirit of Jensen's GB-GA / mol_opt's graph_ga) at the same budget, averaged over 5 GA
seeds. optimum = 1.0 (top-10 mean similarity of 1.0 from the first query).

emin = 0.3, i.e. AUC 0.70 earns 1.0. Exact rediscovery (AUC -> 1.0) is not reachable without
the target, so the practical ceiling is measured instead: the same GA given a 10x budget
(20000 evaluations) ends with a top-10 mean similarity of 0.61 / 0.67 / 0.85 on seeds 1 / 0 / 2,
and the best any pool-only strategy can do with perfect foresight is the pool's own top-10
mean, 0.37-0.45. AUC 0.70 therefore means holding, across the whole 2000-molecule budget, the
similarity level a strong GA only reaches after ten times as many evaluations.

Data: ZINC-250k (249,455 commercially available drug-like molecules), as distributed by
Therapeutics Data Commons `MolGen(name='ZINC')`.
  source: https://dataverse.harvard.edu/api/access/datafile/4170963  (file zinc.tab)
  TDC dataverse: https://doi.org/10.7910/DVN/21LKWG, TDC datasets are CC BY 4.0
  ZINC: Irwin et al., J. Chem. Inf. Model. 2012, 52, 1757 (ZINC is free to use)
The public pool (40k molecules) and the held-back target pool (2k) are disjoint samples of
that file, drawn with a fixed permutation seed so the public files are identical for every
problem seed.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

from .base import Anchors, InvalidSubmission, OracleSpec, Problem, ProblemInfo

RDLogger.DisableLog("rdApp.*")

DATA = Path(__file__).resolve().parents[2] / "data" / "C1"
RAW = DATA / "raw"  # gitignored: the download, kept so fetch_data() is a no-op on rerun
ZINC_URL = "https://dataverse.harvard.edu/api/access/datafile/4170963"

BUDGET = 2_000
MAX_PER_QUERY = 50
TOP_K = 10
POOL_SIZE = 40_000
N_TARGETS = 2_000
SPLIT_SEED = 20_260_101
GA_SEEDS = (0, 1, 2, 3, 4)  # GA run-to-run sd is ~0.05 AUC; 5 runs put the anchor sem near 0.02

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fetch_data() -> None:
    """Download ZINC-250k and split it into the public pool and the held-back targets."""
    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / "zinc.tab"
    if not raw.exists():
        with urllib.request.urlopen(ZINC_URL, timeout=300) as r:  # noqa: S310 (fixed https URL)
            raw.write_bytes(r.read())
    rows = list(csv.reader(io.StringIO(raw.read_text()), delimiter="\t"))
    smiles = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(smiles)

    targets, i = [], 0
    while len(targets) < N_TARGETS and i < len(smiles):  # drug-like, mid-sized, neutral
        s = smiles[i]
        i += 1
        m = Chem.MolFromSmiles(s)
        if m is None or Chem.GetFormalCharge(m) != 0:
            continue
        if 18 <= m.GetNumHeavyAtoms() <= 32 and 250 <= Descriptors.MolWt(m) <= 450:
            targets.append(Chem.MolToSmiles(m))
    pool = [Chem.MolToSmiles(m) for s in smiles[i : i + int(1.05 * POOL_SIZE)]
            if (m := Chem.MolFromSmiles(s)) is not None][:POOL_SIZE]
    if len(targets) < N_TARGETS or len(pool) < POOL_SIZE:
        raise RuntimeError(f"ZINC file yielded {len(targets)} targets / {len(pool)} pool molecules")
    (DATA / "targets.smi").write_text("\n".join(targets) + "\n")
    (DATA / "pool.smi").write_text("\n".join(pool) + "\n")


def _lines(path: Path) -> list[str]:
    if not path.exists():
        fetch_data()
    return path.read_text().split()


@lru_cache(maxsize=1)
def _pool() -> tuple[str, ...]:
    return tuple(_lines(DATA / "pool.smi"))


@lru_cache(maxsize=1)
def _targets() -> tuple[str, ...]:
    return tuple(_lines(DATA / "targets.smi"))


def _fp(mol):
    return _FPGEN.GetFingerprint(mol)


def _auc_top_k(scored: list[tuple[str | None, float | None]], budget: int = BUDGET) -> float:
    """PMO AUC top-10 over a query sequence: see the module docstring for the exact rule.

    `scored` is (fingerprint key, similarity) per queried molecule in order; (None, None)
    for an invalid SMILES. Molecules past `budget` are ignored; a short sequence is padded
    with its final top-10 mean (mol_opt's `finish=True`).
    """
    best: dict[str, float] = {}
    total, running = 0.0, 0.0
    for i in range(budget):
        if i < len(scored):
            key, sim = scored[i]
            if key is not None and sim is not None and sim > best.get(key, -1.0):
                best[key] = sim
                running = sum(sorted(best.values(), reverse=True)[:TOP_K]) / TOP_K
        total += running
    return total / budget


class SealedTargetRediscovery(Problem):
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        targets = _targets()
        self._target_smiles = targets[random.Random(30_000 + seed).randrange(len(targets))]
        self._target_fp = _fp(Chem.MolFromSmiles(self._target_smiles))
        self._log: list[tuple[str | None, float | None]] = []
        self.info = ProblemInfo(
            id="C1",
            domain="chemistry",
            title="Sealed-target rediscovery",
            kind="exact",
            metric="AUC top-10 Tanimoto similarity over the query sequence (higher is better)",
            higher_is_better=True,
            dev_budget=0,
            oracle=OracleSpec(
                budget=BUDGET,
                description=f"Tanimoto similarity to the sealed target; one unit per molecule, "
                            f"at most {MAX_PER_QUERY} molecules per call",
            ),
        )

    def statement(self) -> str:
        return f"""# C1 · Sealed-target rediscovery

One drug-like molecule is sealed on this server. It is **not** in `pool.smi`. Find it, or
get as close to it as you can, with as few molecule evaluations as possible.

**Oracle** (`lab oracle C1 '{{"smiles": ["CCO", "c1ccccc1O", ...]}}'`): returns
`{{"similarity": [...]}}`, the Tanimoto similarity of each molecule to the sealed target
over Morgan fingerprints (radius 2, 2048 bits, RDKit `GetMorganGenerator`). Invalid SMILES
return `null`. At most **{MAX_PER_QUERY} molecules per call**; the whole institute shares
**{BUDGET} molecule evaluations**, and every molecule costs one -- duplicates and invalid
SMILES included.

**Public files:** `pool.smi`, {POOL_SIZE:,} drug-like SMILES from the same library as the
target, one per line. Use them as a starting population; the target itself was held back.

**Metric** (PMO AUC top-10, Gao et al. arXiv:2206.12411): let `g(x)` be the sum of the
similarities of the best **{TOP_K}** distinct fingerprints among the first `x` molecules you
evaluated (molecules with identical fingerprints, e.g. stereoisomers, count once),
divided by {TOP_K}. Your score is `mean(g(1), ..., g({BUDGET}))` -- the area under the
running top-{TOP_K} curve, divided by the budget. `g` never decreases, so an extra query
can only help; if you stop early, `g` is held at its final value to the end of the budget,
so stopping early is not punished either. But a molecule found on query 1 is worth far more
than the same molecule found on query {BUDGET}: **when** you find things is the whole game.
1.0 means {TOP_K} molecules at similarity 1.0 from the first query (strictly, 0.99775, since
putting {TOP_K} molecules in the top {TOP_K} costs {TOP_K} queries). Random sampling from the
pool is the baseline, a graph GA on the same budget is the reference.

**There is no submission file to compute:** your score comes from the oracle log. Submit
anything (e.g. `{{}}`) to register your **claim** of the AUC above, which is what credit is
paid on: `lab eval C1 <file>` reports your AUC so far from the log.
"""

    def public_files(self):
        return {"pool.smi": ("\n".join(_pool()) + "\n").encode()}

    def _score_one(self, smi) -> tuple[str | None, float | None]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        if mol is None:
            return None, None
        fp = _fp(mol)  # dedupe key = the fingerprint itself: stereoisomers etc. share it and count once
        return fp.ToBase64(), float(TanimotoSimilarity(fp, self._target_fp))

    def _score(self, smiles) -> list[tuple[str | None, float | None]]:
        return [self._score_one(s) for s in smiles]

    def oracle(self, query):
        smiles = query.get("smiles") if isinstance(query, dict) else None
        if isinstance(smiles, str):
            smiles = [smiles]
        if not isinstance(smiles, list) or not smiles:
            raise InvalidSubmission("'smiles' must be a non-empty list of SMILES strings")
        if len(smiles) > MAX_PER_QUERY:
            raise InvalidSubmission(f"at most {MAX_PER_QUERY} molecules per call, got {len(smiles)}")
        scored = [self._score_one(s) for s in smiles]
        self._log.extend(scored)
        return {
            "similarity": [None if s is None else round(s, 6) for _, s in scored],
            "_cost": len(smiles),  # one budget unit per molecule, valid or not
        }

    def score_from_oracle_log(self, log) -> float | None:
        scored = [self._score_one(s) for s in _logged_smiles(log)[:BUDGET]] or self._log
        return _auc_top_k(scored)

    def verify(self, submission) -> float:
        """C1 is scored from the oracle log; the submission only carries the claim."""
        if not isinstance(submission, dict):
            raise InvalidSubmission("submission must be a JSON object; C1 is scored from the oracle log")
        return _auc_top_k(self._log)

    def dev_eval(self, submission) -> float:
        # Never let `lab eval` turn into an unbudgeted similarity oracle.
        return self.verify(submission)

    def anchors(self) -> Anchors:
        base, ref = _builder_anchors(self.seed)
        return Anchors(
            baseline=base,
            reference=ref,
            optimum=1.0,
            emin=0.05,  # AUC 0.95 earns 1.0. Was 0.3; a Codex pilot rediscovered a target in ~400 queries (Nov 14)
            ref_kind="B",
            ref_source="builder: random pool sampling (0) / compact graph GA, 5 seeds (0.5), "
                       "both at the same 2000-molecule budget; metric per arXiv:2206.12411",
        )


def _logged_smiles(log) -> list[str]:
    """Pull the queried SMILES, in order, out of whatever shape the server logs."""
    out: list[str] = []
    for entry in log or []:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query") if isinstance(entry.get("query"), dict) else entry
        smiles = query.get("smiles")
        if isinstance(smiles, str):
            smiles = [smiles]
        if isinstance(smiles, list):
            out.extend(s for s in smiles if isinstance(s, str))
    return out


# --------------------------------------------------------------------------- anchors


def _cut(mol, rng) -> list:
    """Split a molecule on one random acyclic single bond; fragments keep a dummy atom."""
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
             and b.GetBeginAtom().GetAtomicNum() > 0 and b.GetEndAtom().GetAtomicNum() > 0]
    if not bonds:
        return []
    frags = Chem.FragmentOnBonds(mol, [rng.choice(bonds)], addDummies=True)
    try:
        return list(Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True))
    except Exception:
        return []


def _join(a, b, rng):
    """Bond the neighbour of a dummy atom in `a` to the neighbour of one in `b`."""
    na = a.GetNumAtoms()
    em = Chem.RWMol(Chem.CombineMols(a, b))
    da = [i for i in range(na) if em.GetAtomWithIdx(i).GetAtomicNum() == 0]
    db = [i for i in range(na, em.GetNumAtoms()) if em.GetAtomWithIdx(i).GetAtomicNum() == 0]
    if not da or not db:
        return None
    ia, ib = rng.choice(da), rng.choice(db)
    ja = [n.GetIdx() for n in em.GetAtomWithIdx(ia).GetNeighbors()]
    jb = [n.GetIdx() for n in em.GetAtomWithIdx(ib).GetNeighbors()]
    if not ja or not jb:
        return None
    em.AddBond(ja[0], jb[0], Chem.BondType.SINGLE)
    for i in sorted([ia, ib], reverse=True):
        em.RemoveAtom(i)
    mol = em.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol if 5 <= mol.GetNumHeavyAtoms() <= 60 else None


@lru_cache(maxsize=1)
def _fragments() -> tuple:
    """Fragment library: one random cut of each of 8000 pool molecules."""
    rng = random.Random(7)
    frags = []
    for smi in rng.sample(_pool(), 8_000):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frags.extend(f for f in _cut(mol, rng) if 2 <= f.GetNumHeavyAtoms() <= 20)
    return tuple(frags)


def _child(p1, p2, rng):
    """Fragment crossover, then mutation: swap a fragment for one from the library."""
    f1, f2 = _cut(p1, rng), _cut(p2, rng)
    if not f1 or not f2:
        return None
    child = _join(rng.choice(f1), rng.choice(f2), rng)
    if child is not None and rng.random() < 0.3:
        parts = _cut(child, rng)
        if parts:
            child = _join(max(parts, key=lambda m: m.GetNumHeavyAtoms()), rng.choice(_fragments()), rng)
    return child


def _ga(target_fp, budget: int, seed: int, pop_size: int = 100, offspring: int = 50):
    """Compact graph GA. Returns the (smiles, similarity) sequence it evaluated, in order."""
    rng = random.Random(seed)
    pool = _pool()
    seq: list[tuple[str | None, float | None]] = []
    seen: set[str] = set()

    def evaluate(mols) -> list[tuple[float, object]]:
        """Spend budget on the unseen molecules; return the (similarity, mol) pairs scored."""
        out = []
        for mol in mols:
            if len(seq) >= budget:
                break
            key = Chem.MolToSmiles(mol)
            if key in seen:
                continue
            seen.add(key)
            sim = float(TanimotoSimilarity(_fp(mol), target_fp))
            seq.append((key, sim))
            out.append((sim, mol))
        return out

    start = [m for s in rng.sample(pool, pop_size) if (m := Chem.MolFromSmiles(s)) is not None]
    population = sorted(evaluate(start), key=lambda t: -t[0])

    while len(seq) < budget:
        weights = [max(s, 1e-6) ** 3 for s, _ in population]  # sharpening: Tanimoto is flat
        kids, tries = [], 0
        while len(kids) < offspring and tries < 20 * offspring:
            tries += 1
            (_, p1), (_, p2) = rng.choices(population, weights=weights, k=2)
            child = _child(p1, p2, rng)
            if child is not None and Chem.MolToSmiles(child) not in seen:
                kids.append(child)
        if not kids:
            kids = [m for s in rng.sample(pool, offspring) if (m := Chem.MolFromSmiles(s)) is not None]
        population = sorted(population + evaluate(kids), key=lambda t: -t[0])[:pop_size]
    return seq


@lru_cache(maxsize=64)
def _builder_anchors(seed: int) -> tuple[float, float]:
    cache = DATA / f"anchors_seed{seed}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["baseline"], d["reference"]
    p = SealedTargetRediscovery(seed)
    rng = random.Random(900 + seed)
    draw = rng.sample(_pool(), BUDGET)
    base = _auc_top_k([p._score_one(s) for s in draw])
    refs = [_auc_top_k(_ga(p._target_fp, BUDGET, 1000 * seed + g)) for g in GA_SEEDS]
    ref = sum(refs) / len(refs)
    if not ref > base:
        raise RuntimeError(f"C1 seed {seed}: GA reference {ref:.4f} did not beat baseline {base:.4f}")
    cache.write_text(json.dumps({"baseline": base, "reference": ref, "ga_runs": refs}, indent=1))
    return base, ref

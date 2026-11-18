"""Research artifact schema v1 and the live (in-run) checks.

Agents research however they like. What they submit must fit one envelope, so every
contribution is checked the same way:

  kind        finding | null_result | method | hypothesis | note | preregistration
  claim       statement, datasets, optional quantities (name, value, units, ci)
  evidence    exec ids (server-run `lab exec`) that produced the claimed quantities
  test        how the evaluator re-tests the claim on hidden data: a verification profile
  relations   [{target, relation: cites|replicates|extends|corrects|refutes, used: [blob ids]}]
  prespec     id of a preregistration artifact (required for null results)

Verification profiles (lab/verify.py runs them after the run ends):
  prediction   test writes $OUT_DIR/predictions.csv (id, pred) for $EVAL_DIR/<dataset>/inputs.csv;
               the server scores it against hidden labels and the environment's baseline
  comparison   test writes predictions_a.csv and predictions_b.csv; the server tests a vs b
  equivalence  as comparison, plus prespecified bounds; the server runs TOST
  custom       anything else: re-execution only, then the auditor (lower evidence tier)
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Kind = Literal["finding", "null_result", "method", "hypothesis", "note", "preregistration"]
Relation = Literal["cites", "replicates", "extends", "corrects", "refutes"]
Profile = Literal["prediction", "comparison", "equivalence", "custom"]
CREDITED = ("finding", "null_result", "method", "hypothesis")


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: float
    units: str = ""
    ci: tuple[float, float] | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(min_length=20, description="one falsifiable sentence")
    datasets: list[str] = Field(default_factory=list, description="dataset ids from `lab datasets`")
    quantities: list[Quantity] = Field(default_factory=list, description="numbers the claim rests on; each must appear in an evidence exec's metrics.json")


class Test(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Profile
    code: str = Field(description="blob id of the test code (`lab put DIR`)")
    cmd: str = Field(description="run in /work; public data at $DATA_DIR, hidden inputs at $EVAL_DIR, write to $OUT_DIR")
    inputs: dict[str, str] = Field(default_factory=dict, description="NAME -> blob id, mounted at $IN_DIR/NAME (e.g. another agent's outputs)")
    dataset: str = ""
    better: Literal["a", "b"] = Field("a", description="comparison: which arm the claim says is better")
    bounds: tuple[float, float] | None = Field(None, description="equivalence: the smallest effect that would matter, as [low, high] on the metric")


class Rel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    relation: Relation
    used: list[str] = Field(default_factory=list, description="blob ids of the target's outputs or code this work consumed")


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Kind
    title: str = Field(min_length=5, max_length=200)
    summary: str = Field("", max_length=20_000)
    claim: Claim | None = None
    test: Test | None = None
    evidence: list[str] = Field(default_factory=list)
    relations: list[Rel] = Field(default_factory=list)
    prespec: str | None = None
    code: list[str] = Field(default_factory=list, description="blob ids of released code")
    references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parts(self):
        if self.kind == "note":
            return self
        if self.kind == "preregistration":
            if self.test is None:
                raise ValueError("a preregistration needs the test it commits to")
            return self
        if self.claim is None or self.test is None:
            raise ValueError(f"a {self.kind} needs a claim and a test")
        if self.kind in ("finding", "null_result", "method") and not self.evidence:
            raise ValueError(f"a {self.kind} needs evidence: at least one exec id")
        if self.test.profile in ("prediction", "comparison", "equivalence") and not self.test.dataset:
            raise ValueError(f"profile {self.test.profile} needs test.dataset")
        if self.test.profile == "equivalence" and not self.test.bounds:
            raise ValueError("an equivalence test needs bounds")
        if self.kind == "null_result":
            if self.test.profile != "equivalence":
                raise ValueError("a null result is verified by equivalence testing (profile 'equivalence')")
            if not self.prespec:
                raise ValueError("a null result needs a preregistration (prespec) submitted before its evidence")
        return self


def value_matches(claimed: float, recorded: float) -> bool:
    """Claims may round; accept agreement to 3 significant figures."""
    if not (math.isfinite(claimed) and math.isfinite(recorded)):
        return False
    return math.isclose(claimed, recorded, rel_tol=5e-3, abs_tol=1e-9)


def numbers(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield float(x)
    elif isinstance(x, dict):
        for v in x.values():
            yield from numbers(v)
    elif isinstance(x, list):
        for v in x:
            yield from numbers(v)


def check_live(art: Artifact, execs: dict[str, dict], arts: dict[str, dict], now: float) -> dict:
    """Checks that run at submission and give the agent feedback. No hidden data involved.
    `execs`: this run's exec records (with "ts"); `arts`: this run's artifacts."""
    checks, why = {}, []
    if art.kind == "note":
        return {"ok": True, "checks": {}, "why": [], "fabrication": False}
    ev = [execs.get(e) for e in art.evidence]
    checks["evidence_exists"] = all(e is not None for e in ev)
    if not checks["evidence_exists"]:
        why.append("unknown exec id in evidence")
    ev = [e for e in ev if e is not None]
    checks["evidence_succeeded"] = all(e["exit_code"] == 0 for e in ev)
    if not checks["evidence_succeeded"]:
        why.append("an evidence exec failed")
    fabricated = False
    if art.claim and art.claim.quantities:
        def recorded(q, pool):
            return [e for e in pool if any(value_matches(q.value, n) for n in numbers(e.get("metrics")))]
        missing = [q for q in art.claim.quantities if not recorded(q, ev)]
        checks["quantities_recorded"] = not missing
        invented = [q.name for q in missing if not recorded(q, list(execs.values()))]
        miscited = {q.name: [e["id"] for e in recorded(q, list(execs.values()))][:3] for q in missing if q.name not in invented}
        if invented:  # a number no recorded run of this institute ever produced
            fabricated = True
            why.append(f"fabrication: claimed quantities appear in no recorded run: {', '.join(invented)}")
        if miscited:
            why.append("mis-cited evidence: " + "; ".join(f"{k} is recorded in {', '.join(v)} but that exec is not listed as evidence"
                                                          for k, v in miscited.items()))
    if art.test:
        from .executor import get_blob
        missing_blobs = []
        for blob in [art.test.code, *art.test.inputs.values()]:
            try:
                get_blob(blob)
            except KeyError:
                missing_blobs.append(blob[:12])
        checks["test_files_uploaded"] = not missing_blobs
        if missing_blobs:
            why.append(f"test refers to files never uploaded (use the code blob id from lab exec or lab put): {', '.join(missing_blobs)}")
    checks["targets_exist"] = all(r.target in arts for r in art.relations)
    if not checks["targets_exist"]:
        why.append("unknown artifact id in relations")
    consumed = set(art.test.inputs.values()) | {art.test.code} if art.test else set()
    for e in ev:
        consumed |= set(e.get("inputs", {}).values()) | {e.get("code")}
    unused = [r.target for r in art.relations if r.used and not set(r.used) <= consumed]
    checks["used_inputs_consumed"] = not unused
    if unused:
        why.append(f"relations claim use of files no evidence exec consumed: {', '.join(unused)}")
    if art.prespec:
        pre = arts.get(art.prespec)
        ok = bool(pre) and pre["kind"] == "preregistration" and all(pre["ts"] < e["ts"] for e in ev)
        if ok and art.test and art.test.bounds:
            ok = tuple(pre.get("test", {}).get("bounds") or ()) == tuple(art.test.bounds)
        checks["prespec_before_evidence"] = ok
        if not ok:
            why.append("preregistration missing, not a preregistration, submitted after the evidence, or with different bounds")
    return {"ok": all(checks.values()), "checks": checks, "why": why, "fabrication": fabricated}

"""Problem interface. Every problem lives on the lab server; agents only ever see
`statement()` and `public_files()`. Hidden data, secret seeds and verifiers stay here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


class InvalidSubmission(ValueError):
    """Raised when a submission or oracle query is malformed or infeasible."""


@dataclass(frozen=True)
class Anchors:
    """Raw-metric anchors for the reference-anchored log scale.

    baseline  -> score 0.0
    reference -> score 0.5
    optimum   -> m*, the target the error e(m) = |m - m*| is measured against
                 (a known optimum, a proven bound, or 0 for a loss)
    emin      -> error that earns 1.0 (verifier tolerance / noise floor);
                 None means mirror the lower segment: emin = eref**2 / e0
    """

    baseline: float
    reference: float
    optimum: float
    emin: float | None = None
    ref_kind: Literal["P", "B"] = "B"  # P = published, B = builder-computed
    ref_source: str = ""


@dataclass
class OracleSpec:
    budget: int = 0  # queries per run, shared by the whole institute
    description: str = ""


@dataclass
class ProblemInfo:
    id: str
    domain: Literal["math", "physics", "chemistry", "biology", "materials", "ml"]
    title: str
    kind: Literal["exact", "generalization"]
    metric: str
    higher_is_better: bool
    dev_budget: int = 0  # server-side dev evaluations per run (generalization problems)
    oracle: OracleSpec = field(default_factory=OracleSpec)
    max_submissions: int = 5


class Problem(ABC):
    """One open problem instance. `seed` selects the hidden instance for sealed problems."""

    info: ProblemInfo

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def statement(self) -> str:
        """Markdown shown to agents: task, submission JSON format, oracle usage, baseline."""

    def public_files(self) -> dict[str, bytes]:
        """Files copied to /lab/problems/<id>/ in the sandbox (training data, inputs)."""
        return {}

    @abstractmethod
    def anchors(self) -> Anchors:
        ...

    @abstractmethod
    def verify(self, submission: dict[str, Any]) -> float:
        """Raw metric on the hidden test (or exact objective). Raise InvalidSubmission."""

    def dev_eval(self, submission: dict[str, Any]) -> float:
        """Raw metric on the server-held dev split. Exact problems verify directly."""
        if self.info.kind == "exact":
            return self.verify(submission)
        raise NotImplementedError

    def oracle(self, query: dict[str, Any]) -> dict[str, Any]:
        """Answer one budgeted experiment. Only for problems with an oracle."""
        raise InvalidSubmission("this problem has no oracle")

    def score_from_oracle_log(self, log: list[dict[str, Any]]) -> float | None:
        """Raw metric computed from the oracle log instead of a submission (e.g. C1)."""
        return None

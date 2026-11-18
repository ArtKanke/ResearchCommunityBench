"""The board: problem id -> constructor(seed). Edit BOARD to change what institutes face."""

from __future__ import annotations

import importlib
from functools import lru_cache

from .base import Problem

# id -> "module:Class[:arg]"  (arg is passed as the second constructor argument)
REGISTRY = {
    "M1": "m1_autocorr:AutocorrelationM1",
    "M2": "m2_circles:CirclePacking26",
    "P1": "p_sealed:SealedSystem:P1",
    "P2": "p_sealed:SealedSystem:P2",
    "C1": "c1_rediscovery:SealedTargetRediscovery",
    "C2": "c2_solubility:AqSolubility",
    "B1": "b1_ehrlich:EhrlichDesign",
    "B2": "b2_proteingym:ProteinFitness",
    "X1": "x1_bandgap:BandGap",
    "X2": "x2_lj_cluster:BinaryLJCluster",
}

DEFAULT_BOARD = ["M1", "M2", "P1", "P2", "C1", "C2", "B1", "B2", "X1", "X2"]


def available() -> list[str]:
    out = []
    for pid in REGISTRY:
        try:
            _cls(pid)
            out.append(pid)
        except Exception:  # a module that fails to import is simply off the board
            pass
    return out


def _cls(pid):
    mod, cls, *arg = REGISTRY[pid].split(":")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), cls), arg


@lru_cache(maxsize=256)
def get(pid: str, seed: int) -> Problem:
    cls, arg = _cls(pid)
    return cls(seed, *arg)

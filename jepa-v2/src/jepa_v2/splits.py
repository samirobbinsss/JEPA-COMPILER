"""Deterministic, disjoint corpus splitting — the anti-leakage backbone.

Copied as-is from v1 (../jepa-ir/src/jepa_ir/data/splits.py). Every program is
assigned to a pool by a stable hash of its NAME, so the assignment is identical
across scripts and runs, with no overlap. This guarantees the encoder never
trains on programs used to evaluate it, and a held-out pool is seen by nothing.

Pools (by default):
  ENCODER   : trains the GNN encoder (self-supervised JEPA)
  PREDICTOR : reserved (disjoint) — kept for parity with v1 / future probes
  HELDOUT   : final evaluation, touched by nothing

Within a pool, `subsplit(name, ratios)` gives a further deterministic
train/val/test partition (again by hash, so stable & disjoint).
"""
from __future__ import annotations

import hashlib

POOLS = ("encoder", "predictor", "heldout")
# default pool proportions (must sum to 1.0)
POOL_RATIOS = {"encoder": 0.45, "predictor": 0.45, "heldout": 0.10}


def _unit_hash(name: str, salt: str = "") -> float:
    """Stable hash of a name -> float in [0, 1). Deterministic across processes
    (uses hashlib, NOT Python's salted hash())."""
    h = hashlib.sha1(f"{salt}:{name}".encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def _orig_name(filename: str) -> str:
    """Strip a sampler's 'NNNNNNN_' index prefix so the SAME source maps to the
    same pool whether it came from sample_100k, sample_200k, etc."""
    base = filename.rsplit("/", 1)[-1]
    if "_" in base and base.split("_", 1)[0].isdigit():
        base = base.split("_", 1)[1]
    return base


def pool_of(filename: str) -> str:
    """Return which top-level pool a program belongs to (stable by name)."""
    u = _unit_hash(_orig_name(filename), salt="pool")
    acc = 0.0
    for p in POOLS:
        acc += POOL_RATIOS[p]
        if u < acc:
            return p
    return POOLS[-1]


def in_pool(filename: str, pool: str) -> bool:
    return pool_of(filename) == pool


def subsplit(filename: str, ratios=(0.70, 0.15, 0.15)) -> str:
    """Within a pool, assign to 'train'/'val'/'test' deterministically by name."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    u = _unit_hash(_orig_name(filename), salt="subsplit")
    tr, va, _te = ratios
    if u < tr:
        return "train"
    if u < tr + va:
        return "val"
    return "test"

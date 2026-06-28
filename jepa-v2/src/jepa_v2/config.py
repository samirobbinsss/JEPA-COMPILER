"""Configuration dataclasses for jepa-v2.

Kept deliberately small and explicit — every number that matters to a run lives
here so checkpoints and logs are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The optimization levels we compile every source at. The whole point of v2 is
# that these produce DIFFERENT graphs (gated by the ExeBench probe).
OPT_LEVELS = ("-O0", "-O1", "-O2", "-O3")

# z_speed groups -O levels into "speed classes" (positives share a class).
# DEFAULT = 4 classes (identity): treat every -O level as its own class.
# The Step-1 gate (docs/results_gate_exebench.md) found O2≈O3 graphs identical on
# ExeBench, so a 4-class objective will see O2/O3 merge on its own; switch to
# (0, 1, 2, 2) to merge them explicitly, or (0, 1, 1, 1) for opt/no-opt.
SPEED_GROUPS = (0, 1, 2, 3)


@dataclass
class ModelConfig:
    # node feature vocab (built from ProgramML node `text`); 0 reserved for <unk>
    vocab_size: int = 8192
    node_emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 6
    # the embedding is FACTORED: total = sem_dim + speed_dim
    sem_dim: int = 96      # z_sem — invariant to -O
    speed_dim: int = 32    # z_speed — invariant to program, varies by -O
    num_edge_types: int = 3  # control, data, call (ProgramML flow)
    dropout: float = 0.0

    @property
    def emb_dim(self) -> int:
        return self.sem_dim + self.speed_dim


@dataclass
class VICRegConfig:
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    eps: float = 1e-4


@dataclass
class LossConfig:
    # VICReg applied separately to each block
    vicreg_sem: VICRegConfig = field(default_factory=VICRegConfig)
    vicreg_speed: VICRegConfig = field(default_factory=VICRegConfig)
    # weight of each invariance objective
    sem_weight: float = 1.0     # pull the 4 -O of one source together (in z_sem)
    speed_weight: float = 1.0   # pull same -O across sources together (in z_speed)
    # cross-decorrelation: force z_sem ⟂ z_speed (the disentanglement term)
    cross_decorr_weight: float = 1.0


@dataclass
class DataConfig:
    # which ExeBench split to build the corpus from
    split: str = "train_real_compilable"
    # drop programs whose -O0 graph is smaller than this (degenerate / decl-only).
    # The gate showed the O1/O2 signal only appears above ~50 nodes.
    min_nodes: int = 16
    # a program is kept only if ALL opt levels compile to a valid graph
    require_all_levels: bool = True
    vocab_size: int = 8192


@dataclass
class TrainConfig:
    batch_programs: int = 64   # programs per batch; each contributes 4 -O views
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 50
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    num_workers: int = 8
    seed: int = 0

"""Factored self-supervised loss: z = [z_sem | z_speed].

For a batch of P programs x V opt-level views (rows tagged with `prog` and `lvl`):

  L_sem   : in z_sem, pull together the V views of the SAME program (invariant to
            -O) — implemented as MSE of each view to its program centroid — plus
            VICReg variance+covariance over the whole z_sem batch (anti-collapse).
  L_speed : in z_speed, pull together views of the SAME -O level across DIFFERENT
            programs (invariant to the program) — MSE to per-level centroid — plus
            VICReg variance+covariance over z_speed.
  L_cross : minimize the cross-covariance between z_sem and z_speed dims. This is
            the disentanglement term that forces z_sem ⟂ z_speed.

Centroid-MSE is the group generalization of VICReg's pairwise invariance: with
exactly 2 views it equals 1/2 * MSE(view_a, view_b), but it also handles V>2 and
ragged groups cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import scatter

from .config import SPEED_GROUPS, LossConfig, VICRegConfig
from .vicreg import covariance_term, cross_covariance_term, variance_term


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: dict[str, float]


def _group_invariance(z: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Mean squared distance of each row to its group's centroid (detached center
    is NOT used — we want gradients to pull views together symmetrically)."""
    num = int(group.max().item()) + 1 if group.numel() else 0
    if num == 0:
        return z.new_zeros(())
    centroids = scatter(z, group, dim=0, dim_size=num, reduce="mean")
    # mean over BOTH rows and feature dims (VICReg convention). Summing over dims
    # would hide a factor of D in the effective sim_coeff and drive collapse.
    return ((z - centroids[group]) ** 2).mean()


def _block_vicreg(z: torch.Tensor, group: torch.Tensor, cfg: VICRegConfig):
    inv = _group_invariance(z, group)
    var = variance_term(z, cfg.eps)
    cov = covariance_term(z)
    total = cfg.sim_coeff * inv + cfg.std_coeff * var + cfg.cov_coeff * cov
    return total, inv, var, cov


def _remap_speed(lvl: torch.Tensor) -> torch.Tensor:
    """Map -O level index -> speed class via config.SPEED_GROUPS (e.g. merge O2/O3)."""
    table = torch.tensor(SPEED_GROUPS, device=lvl.device, dtype=torch.long)
    return table[lvl]


def factored_loss(
    z_sem: torch.Tensor,
    z_speed: torch.Tensor,
    prog: torch.Tensor,
    lvl: torch.Tensor,
    cfg: LossConfig,
) -> LossOutput:
    """z_sem:[B,Ds] z_speed:[B,Dp]; prog,lvl:[B] integer labels."""
    speed_cls = _remap_speed(lvl)

    sem_total, sem_inv, sem_var, sem_cov = _block_vicreg(z_sem, prog, cfg.vicreg_sem)
    spd_total, spd_inv, spd_var, spd_cov = _block_vicreg(z_speed, speed_cls, cfg.vicreg_speed)
    cross = cross_covariance_term(z_sem, z_speed)

    total = (
        cfg.sem_weight * sem_total
        + cfg.speed_weight * spd_total
        + cfg.cross_decorr_weight * cross
    )
    parts = {
        "total": float(total.detach()),
        "sem_inv": float(sem_inv.detach()),
        "sem_var": float(sem_var.detach()),
        "sem_cov": float(sem_cov.detach()),
        "speed_inv": float(spd_inv.detach()),
        "speed_var": float(spd_var.detach()),
        "speed_cov": float(spd_cov.detach()),
        "cross": float(cross.detach()),
    }
    return LossOutput(total=total, parts=parts)

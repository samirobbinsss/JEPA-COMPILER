"""GNN encoder with a FACTORED head: graph -> z = [z_sem | z_speed].

A single 3-relation GNN trunk (control / data / call) produces graph-level
features; two separate projection heads split the embedding into:

  z_sem   — meant to be invariant to the -O level (what the code does)
  z_speed — meant to be invariant to the program (the optimization profile)

The factorization is enforced by the LOSS (see loss.py), not by the architecture
alone — the two heads only give the optimizer two places to put the two kinds of
information. Trained from scratch; no pretrained weights.

ProgramML graph -> PyG:
  node feature: an integer token id from the node `text` vocab (see data.py)
  edge_index per flow type: control(0), data(1), call(2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

from .config import ModelConfig

# ProgramML edge flow -> relation index (kept here so model + data agree).
FLOW_TO_REL = {"control": 0, "data": 1, "call": 2}
REL_NAMES = ("control", "data", "call")


class StructuralPE(nn.Module):
    """Per-relation log-degree encoding projected into hidden_dim."""

    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        self.lin = nn.Linear(2 * num_relations, hidden_dim)

    def forward(self, num_nodes, edge_indices, device):  # noqa: ANN001
        feats = []
        for ei in edge_indices:
            if ei.numel() == 0:
                indeg = torch.zeros(num_nodes, device=device)
                outdeg = torch.zeros(num_nodes, device=device)
            else:
                outdeg = degree(ei[0], num_nodes=num_nodes).to(device)
                indeg = degree(ei[1], num_nodes=num_nodes).to(device)
            feats.append(torch.log1p(indeg))
            feats.append(torch.log1p(outdeg))
        return self.lin(torch.stack(feats, dim=1))


def _proj_head(in_dim: int, out_dim: int) -> nn.Module:
    """VICReg-style expander/projector: small MLP with BN."""
    hidden = max(out_dim * 2, 128)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class FactoredEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_rel = cfg.num_edge_types

        self.node_embed = nn.Embedding(cfg.vocab_size, cfg.node_emb_dim, padding_idx=0)
        self.input_proj = nn.Linear(cfg.node_emb_dim, cfg.hidden_dim)
        self.pe = StructuralPE(cfg.hidden_dim, self.num_rel)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(
                nn.ModuleList([GraphConv(cfg.hidden_dim, cfg.hidden_dim) for _ in range(self.num_rel)])
            )
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))
        self.dropout = nn.Dropout(cfg.dropout)

        pooled_dim = 2 * cfg.hidden_dim  # mean || max
        # two heads: the factored embedding
        self.head_sem = _proj_head(pooled_dim, cfg.sem_dim)
        self.head_speed = _proj_head(pooled_dim, cfg.speed_dim)

    def _edge_lists(self, data):  # noqa: ANN001
        """Return a list of edge_index tensors, one per relation."""
        eis = []
        for r in range(self.num_rel):
            key = f"edge_index_{r}"
            if hasattr(data, key):
                eis.append(getattr(data, key))
            else:
                eis.append(torch.empty(2, 0, dtype=torch.long, device=data.x.device))
        return eis

    def trunk(self, data) -> torch.Tensor:  # noqa: ANN001
        """Per-node hidden states after message passing -> [N, H]."""
        h = self.input_proj(self.node_embed(data.x.view(-1)))
        edge_indices = self._edge_lists(data)
        h = h + self.pe(h.size(0), edge_indices, h.device)
        for convs, norm in zip(self.convs, self.norms):
            msg = torch.zeros_like(h)
            for r, conv in enumerate(convs):
                ei = edge_indices[r]
                if ei.numel() > 0:
                    msg = msg + conv(h, ei)
            h = norm(h + self.dropout(F.relu(msg)))
        return h

    def forward(self, data):  # noqa: ANN001
        """Return (z_sem, z_speed), UN-normalized. Shapes [B, sem], [B, speed].

        VICReg's variance term needs the raw per-dim scale (it hinges std>=1), so we
        deliberately do NOT L2-normalize here. The eval normalizes for cosine.
        """
        h = self.trunk(data)
        b = data.batch if hasattr(data, "batch") and data.batch is not None else \
            torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], dim=1)
        z_sem = self.head_sem(pooled)
        z_speed = self.head_speed(pooled)
        return z_sem, z_speed

    @torch.no_grad()
    def embed(self, data):  # noqa: ANN001
        """Inference: the deliverable embedding z = [z_sem | z_speed]."""
        self.eval()
        z_sem, z_speed = self.forward(data)
        return torch.cat([z_sem, z_speed], dim=1)

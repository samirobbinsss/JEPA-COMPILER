#!/usr/bin/env python3
"""Step 6 — evaluate the disentanglement (the result to show), PCA-centric.

The deliverable is the encoder and how cleanly z = [z_sem | z_speed] factorizes:

  z_sem   should be INVARIANT to -O  (cluster by PROGRAM, not by -O level)
  z_speed should be INVARIANT to the program (cluster by -O LEVEL, not by program)

We quantify and VISUALIZE this with PCA:
  * PCA(z_speed) colored by -O level  -> should separate into level clusters
  * PCA(z_sem)   colored by program   -> should separate into program clusters
  * the OFF-target colorings (PCA(z_sem) by -O, PCA(z_speed) by program) should
    look mixed — that is the disentanglement.
  * silhouette score of each (space, label) pair makes the picture a number;
  * cosine intra/inter tables give the classic JEPA invariance summary.

Run on the pod:
  scripts/pod.sh run 'python3 scripts/eval_disentangle.py --ckpt checkpoints/encoder.pt --cache data/cache'
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jepa_v2.config import ModelConfig, OPT_LEVELS  # noqa: E402
from jepa_v2.data import (  # noqa: E402
    ProgramDataset, Vocab, collate_programs, load_cache,
)
from jepa_v2.model import FactoredEncoder  # noqa: E402


@torch.no_grad()
def embed_all(model, programs, dev, batch_programs=32):
    """Return z_sem, z_speed (numpy) and prog/lvl label arrays over all views."""
    from torch.utils.data import DataLoader

    loader = DataLoader(ProgramDataset(programs), batch_size=batch_programs,
                        shuffle=False, collate_fn=collate_programs, drop_last=False)
    sem, spd, progs, lvls = [], [], [], []
    base = 0
    model.eval()
    for batch in loader:
        batch = batch.to(dev)
        zs, zp = model(batch)
        sem.append(zs.cpu())
        spd.append(zp.cpu())
        progs.append(batch.prog.cpu() + base)   # globally-unique program ids
        lvls.append(batch.lvl.cpu())
        base += int(batch.prog.max().item()) + 1
    return (torch.cat(sem).numpy(), torch.cat(spd).numpy(),
            torch.cat(progs).numpy(), torch.cat(lvls).numpy())


def latent_usage(z):
    """How many dims the block actually uses: effective rank (exp-entropy of the
    singular-value spectrum) and #PCA components needed for 90% / 99% variance."""
    import numpy as np

    zc = z - z.mean(0, keepdims=True)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    var = (s ** 2)
    cum = np.cumsum(var) / var.sum()
    return {
        "allocated_dims": int(z.shape[1]),
        "effective_rank": round(eff_rank, 2),
        "dims_for_90pct_var": int((cum < 0.90).sum() + 1),
        "dims_for_99pct_var": int((cum < 0.99).sum() + 1),
    }


def cosine_intra_inter(z, same_label, n_pairs=200_000, seed=0):
    """Mean cosine for pairs sharing a label (intra) vs not (inter)."""
    import numpy as np

    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    m = zn.shape[0]
    i = rng.integers(0, m, n_pairs)
    j = rng.integers(0, m, n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    cos = (zn[i] * zn[j]).sum(1)
    intra = same_label[i] == same_label[j]
    return float(cos[intra].mean()), float(cos[~intra].mean())


def pca_panels(sem, spd, progs, lvls, out_dir):
    """4-panel PCA figure + per-panel silhouette scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    sem2 = PCA(n_components=2).fit(sem)
    spd2 = PCA(n_components=2).fit(spd)
    sem_xy, spd_xy = sem2.transform(sem), spd2.transform(spd)

    # -O is ORDINAL (optimization intensity) -> a discrete blue->red sequence,
    # one crisp color per level, with a banded colorbar labeled -O0..-O3.
    n_lvl = len(OPT_LEVELS)
    level_colors = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"][:n_lvl]
    level_cmap = ListedColormap(level_colors)
    level_norm = BoundaryNorm([i - 0.5 for i in range(n_lvl + 1)], n_lvl)
    # programs are CATEGORICAL with no order -> a high-variety cyclic map.
    prog_cmap = "gist_rainbow"

    panels = [
        ("z_speed PCA — colored by -O (TARGET: cluster)", spd_xy, lvls, "level"),
        ("z_sem PCA — colored by program (TARGET: cluster)", sem_xy, progs, "program"),
        ("z_sem PCA — colored by -O (want: MIXED)", sem_xy, lvls, "level"),
        ("z_speed PCA — colored by program (want: MIXED)", spd_xy, progs, "program"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (title, xy, lab, _kind) in zip(axes.flat, panels):
        if _kind == "level":
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=lab, cmap=level_cmap,
                            norm=level_norm, s=10, alpha=0.75,
                            edgecolors="white", linewidths=0.15)
            cb = fig.colorbar(sc, ax=ax, ticks=range(n_lvl), label="-O")
            cb.ax.set_yticklabels(list(OPT_LEVELS))
        else:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=lab, cmap=prog_cmap, s=8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_disentangle.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    def sil(xy, lab):
        try:
            return round(float(silhouette_score(xy, lab)), 3)
        except Exception:  # noqa: BLE001  (1 cluster, etc.)
            return None

    return path, {
        "explained_var_z_speed": [round(float(v), 3) for v in spd2.explained_variance_ratio_],
        "explained_var_z_sem": [round(float(v), 3) for v in sem2.explained_variance_ratio_],
        # on-target: high is good
        "silhouette_speed_by_level": sil(spd_xy, lvls),
        "silhouette_sem_by_program": sil(sem_xy, progs),
        # off-target: LOW (near 0) is good = disentangled
        "silhouette_sem_by_level": sil(sem_xy, lvls),
        "silhouette_speed_by_program": sil(spd_xy, progs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate disentanglement (PCA)")
    ap.add_argument("--ckpt", default="checkpoints/encoder.pt")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--split", default="test", help="subsplit to eval (train/val/test/all)")
    ap.add_argument("--max-programs", type=int, default=400)
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    programs, names = load_cache(os.path.join(args.cache, "cache.pt"))
    subsplits = torch.load(os.path.join(args.cache, "subsplits.pt"), weights_only=False)
    if args.split != "all":
        idx = [i for i, s in enumerate(subsplits) if s == args.split]
        programs = [programs[i] for i in idx] or programs
    programs = programs[: args.max_programs]

    blob = torch.load(args.ckpt, weights_only=False)
    mcfg = ModelConfig(**blob["model_cfg"]) if "model_cfg" in blob else \
        ModelConfig(vocab_size=blob.get("vocab_size", Vocab.load(
            os.path.join(args.cache, "vocab.json")).size))
    model = FactoredEncoder(mcfg).to(dev)
    model.load_state_dict(blob["model"])

    sem, spd, progs, lvls = embed_all(model, programs, dev)
    print(f"EVAL::EMBED views={sem.shape[0]} programs={len(programs)} "
          f"sem_dim={sem.shape[1]} speed_dim={spd.shape[1]}", flush=True)

    sem_intra, sem_inter = cosine_intra_inter(sem, progs)   # z_sem by program
    spd_intra, spd_inter = cosine_intra_inter(spd, lvls)    # z_speed by level
    fig_path, pca_stats = pca_panels(sem, spd, progs, lvls, args.out)

    report = {
        "z_sem_usage": latent_usage(sem),
        "z_speed_usage": latent_usage(spd),
        "z_sem_cos_intra_program": round(sem_intra, 3),
        "z_sem_cos_inter_program": round(sem_inter, 3),
        "z_sem_gap": round(sem_intra - sem_inter, 3),
        "z_speed_cos_intra_level": round(spd_intra, 3),
        "z_speed_cos_inter_level": round(spd_inter, 3),
        "z_speed_gap": round(spd_intra - spd_inter, 3),
        **pca_stats,
        "figure": fig_path,
    }
    with open(os.path.join(args.out, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("EVAL::REPORT " + json.dumps(report, indent=2))
    print("EVAL::DONE")


if __name__ == "__main__":
    main()

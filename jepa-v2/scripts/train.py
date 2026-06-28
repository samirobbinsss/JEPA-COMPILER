#!/usr/bin/env python3
"""Step 5 — train the FactoredEncoder with the factored self-supervised loss.

  batch = `batch_programs` programs, each contributing 4 -O views;
  loss  = sem (pull a program's -O views together in z_sem)
        + speed (pull same-level views across programs together in z_speed)
        + cross-decorrelation (force z_sem ⟂ z_speed).

Logs the sub-losses plus anti-collapse diagnostics (emb_std, effective rank). A
healthy run keeps emb_std ~1.0; a collapse drives it to 0.

Run on the pod:
  scripts/pod.sh run 'python3 scripts/train.py --cache data/cache --smoke'
  scripts/pod.sh run 'python3 scripts/train.py --cache data/cache --epochs 50'
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jepa_v2.config import LossConfig, ModelConfig, TrainConfig  # noqa: E402
from jepa_v2.data import (  # noqa: E402
    ProgramDataset, Vocab, load_cache, make_loader,
)
from jepa_v2.loss import factored_loss  # noqa: E402
from jepa_v2.model import FactoredEncoder  # noqa: E402


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy of normalized singular values) — soft count of used dims."""
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z.float())
    p = s / (s.sum() + 1e-9)
    ent = -(p * (p + 1e-9).log()).sum()
    return float(ent.exp())


def emb_std(z: torch.Tensor) -> float:
    return float(z.std(dim=0).mean())


def filter_split(programs, names, subsplits, want: str):
    keep = [i for i, s in enumerate(subsplits) if s == want]
    return [programs[i] for i in keep], [names[i] for i in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FactoredEncoder")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-programs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny overfit run: 8 programs, 60 steps, assert no NaN")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    programs, names = load_cache(os.path.join(args.cache, "cache.pt"))
    subsplits = torch.load(os.path.join(args.cache, "subsplits.pt"), weights_only=False)
    vocab = Vocab.load(os.path.join(args.cache, "vocab.json"))
    tr_programs, _ = filter_split(programs, names, subsplits, "train")
    if not tr_programs:
        tr_programs = programs  # tiny cache fallback

    tcfg = TrainConfig()
    if args.smoke:
        tr_programs = tr_programs[:8]
        args.epochs, args.batch_programs = 60, min(8, len(tr_programs))

    ds = ProgramDataset(tr_programs)
    loader = make_loader(ds, args.batch_programs, shuffle=True,
                         num_workers=args.num_workers)

    mcfg = ModelConfig(vocab_size=vocab.size)
    model = FactoredEncoder(mcfg).to(dev)
    lcfg = LossConfig()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=tcfg.weight_decay)

    steps_per_epoch = max(1, len(loader))
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, (tcfg.warmup_epochs if not args.smoke else 1) * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = batch.to(dev)
            z_sem, z_speed = model(batch)
            out = factored_loss(z_sem, z_speed, batch.prog, batch.lvl, lcfg)
            if not torch.isfinite(out.total):
                raise SystemExit(f"TRAIN::NAN at step {step}: {out.parts}")
            opt.zero_grad()
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if step % (5 if args.smoke else 20) == 0 or step == 1:
                d = out.parts
                print(f"TRAIN::STEP ep={epoch} step={step} "
                      f"loss={d['total']:.3f} sem_inv={d['sem_inv']:.3f} "
                      f"speed_inv={d['speed_inv']:.3f} cross={d['cross']:.4f} "
                      f"sem_std={emb_std(z_sem):.3f} speed_std={emb_std(z_speed):.3f} "
                      f"sem_rank={effective_rank(z_sem):.1f} "
                      f"speed_rank={effective_rank(z_speed):.1f} "
                      f"lr={sched.get_last_lr()[0]:.5f}", flush=True)

    ckpt = os.path.join(args.out, "encoder.pt")
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__,
                "vocab_size": vocab.size}, ckpt)
    print(f"TRAIN::DONE steps={step} -> {ckpt}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 2+3 — build the training cache from ExeBench.

For each ExeBench function: compile all 4 -O levels with programl's bundled
clang-10, convert to lightweight (texts, edges), keep only programs that compile
at EVERY level and whose -O0 graph is large enough (the gate showed tiny isolated
functions carry no O1/O2 signal). Then build the node-`text` vocab over the kept
corpus and materialize PyG ProgramData. Saves a single cache + vocab.

Run on the pod:
  scripts/pod.sh put scripts/build_cache.py
  scripts/pod.sh run 'python3 scripts/build_cache.py --n 4000 --split train_real_compilable --out data/cache'
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jepa_v2 import programl_compat  # noqa: F401,E402  (dgl stub before programl)
from jepa_v2 import exebench  # noqa: E402
from jepa_v2.config import OPT_LEVELS  # noqa: E402
from jepa_v2.data import (  # noqa: E402
    Vocab, compile_view, extract, save_cache, view_to_data,
)
from jepa_v2.splits import pool_of, subsplit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ExeBench PyG cache")
    ap.add_argument("--split", default="train_real_compilable")
    ap.add_argument("--n", type=int, default=4000, help="programs to KEEP (target)")
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap ExeBench records scanned (0 = until n kept)")
    ap.add_argument("--min-nodes", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--pool", default="encoder",
                    help="only keep programs in this split pool (encoder/predictor/heldout/all)")
    ap.add_argument("--out", default="data/cache")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # pass 1: compile + extract lightweight views, accumulate (compile once).
    kept: list[dict] = []          # {name, views:[(texts,edges,lvl)]}
    scanned = compiled_fail = degen = pooled_out = 0
    limit = args.max_records or None

    for i, rec in enumerate(exebench.iter_records(args.split, limit=limit)):
        scanned += 1
        name = exebench.record_name(rec, i)
        if args.pool != "all" and pool_of(name) != args.pool:
            pooled_out += 1
            continue
        src = exebench.source_of(rec)
        if src is None:
            continue
        graphs = [compile_view(src, opt, args.timeout) for opt in OPT_LEVELS]
        if any(g is None for g in graphs):
            compiled_fail += 1
            continue
        if graphs[0].number_of_nodes() < args.min_nodes:
            degen += 1
            continue
        views = []
        for lvl, g in enumerate(graphs):
            texts, edges = extract(g)
            views.append((texts, edges, lvl))
        kept.append({"name": name, "views": views})
        if len(kept) % 100 == 0:
            print(f"BUILD::PROGRESS kept={len(kept)} scanned={scanned} "
                  f"fail={compiled_fail} degen={degen} t={time.time()-t0:.0f}s", flush=True)
        if len(kept) >= args.n:
            break

    if not kept:
        raise SystemExit("BUILD::ERROR no programs kept — check split/min-nodes")

    # vocab over all kept node texts
    vocab = Vocab.build((t for p in kept for (t, _e, _l) in p["views"]), args.vocab_size)
    vocab.save(os.path.join(args.out, "vocab.json"))
    print(f"BUILD::VOCAB size={vocab.size}", flush=True)

    # pass 2: materialize ProgramData, attach subsplit
    programs, names, subsplits = [], [], []
    for p in kept:
        views = [view_to_data(t, e, vocab, lvl) for (t, e, lvl) in p["views"]]
        programs.append(views)
        names.append(p["name"])
        subsplits.append(subsplit(p["name"]))

    cache_path = os.path.join(args.out, "cache.pt")
    save_cache(programs, names, cache_path)
    import torch
    torch.save(subsplits, os.path.join(args.out, "subsplits.pt"))

    from collections import Counter
    dist = Counter(subsplits)
    print(f"BUILD::DONE kept={len(programs)} scanned={scanned} fail={compiled_fail} "
          f"degen={degen} pooled_out={pooled_out} subsplits={dict(dist)} "
          f"vocab={vocab.size} t={time.time()-t0:.0f}s -> {cache_path}")


if __name__ == "__main__":
    main()

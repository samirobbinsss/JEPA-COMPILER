#!/usr/bin/env python3
"""Step 1 GATE — does the ProgramML graph differ across -O levels on ExeBench?

The whole v2 premise (`z_speed` separates optimization profiles) dies if clang
saturates early, the way it does on AnghaBench (O2 == O3 for 100% of functions,
see ../jepa-ir/docs/limitation_non_bijective.md). This probe MEASURES it before
we train anything.

Method (per program, all on the pod's bundled clang-10):
  1. take N ExeBench functions (synth_deps + func_def so they compile standalone),
  2. compile each at -O0/-O1/-O2/-O3 via programl.from_cpp(copts=["-Ok"]),
  3. to_networkx -> a canonical SIGNATURE
       (n_nodes, n_edges, sorted multiset of node `text`, sorted multiset of edge
        `flow`)  -> sha1 hash,
  4. on programs that compiled at ALL four levels, report:
       % distinct O0!=O1, O1!=O2, O2!=O3   (O2!=O3 is THE number),
       the histogram of distinct-level partitions (e.g. "O0 | O1=O2=O3"),
       node-count distribution (ExeBench funcs should be bigger than AnghaBench's
       ~22-node median, which is the whole reason to switch corpus).

GATE: aim for >= ~50% distinct O2 vs O3. If ~0%, fall back to whole-program
corpora (cbench/MiBench) or accept z_speed only separating {O0} vs {O1,O2,O3}.

Run on the pod (ProgramML has no arm64 wheel):
  scripts/pod.sh put scripts/probe_exebench.py
  scripts/pod.sh run 'python3 scripts/probe_exebench.py --n 300 --out probe_report.json'
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import tarfile
import time

# Import the dgl stub BEFORE programl (top-level `import programl` pulls in dgl).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jepa_v2.programl_compat  # noqa: F401,E402  (side effect: dgl stub)
import programl as pg  # noqa: E402

OPT_LEVELS = ("-O0", "-O1", "-O2", "-O3")

# ExeBench is shipped as per-split .tar.gz on the HF hub; the loader script is
# broken with modern `datasets` (forces downloading every archive, no streaming).
# We bypass it: download ONE split tarball and read its .jsonl.zst shards directly.
# Each json line is {"meta": {}, "text": {<the real record>}}; the useful fields
# (func_def, real_deps/angha_deps) live under "text".
SPLIT_TARBALL = {
    "test_real": "test_real.tar.gz",
    "valid_real": "valid_real.tar.gz",
    "test_synth": "test_synth.tar.gz",
    "valid_synth": "valid_synth.tar.gz",
    "train_real_compilable": "train_real_compilable.tar.gz",
    "train_synth_compilable": "train_synth_compilable.tar.gz",
}


def load_exebench_rows(n: int, split: str):
    """Yield up to n ExeBench `text` records from a split tarball (direct read)."""
    import zstandard as zstd
    from huggingface_hub import hf_hub_download

    fname = SPLIT_TARBALL.get(split, split)
    path = hf_hub_download("jordiae/exebench", fname, repo_type="dataset")
    print(f"PROBE::SPLIT {split} -> {fname}", flush=True)
    dctx = zstd.ZstdDecompressor()
    yielded = 0
    with tarfile.open(path, "r:*") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".jsonl.zst")):
                continue
            raw = dctx.stream_reader(tf.extractfile(m)).read()
            for line in raw.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text")
                if isinstance(text, dict):
                    yield text
                    yielded += 1
                    if yielded >= n:
                        return


def source_of(rec: dict) -> str | None:
    """Build standalone-compilable C from an ExeBench `text` record.

    deps (real_deps for the real splits, angha_deps for the synth splits) provide
    the includes/typedefs so func_def compiles in isolation.
    """
    func = rec.get("func_def")
    if not isinstance(func, str) or not func.strip():
        return None
    deps = rec.get("real_deps") or rec.get("angha_deps") or ""
    if not isinstance(deps, str):
        deps = ""
    return f"{deps}\n{func}\n"


def signature(g) -> str:
    """Canonical hash of a ProgramML networkx graph (structure + labels + flows)."""
    texts = collections.Counter(d.get("text", "?") for _, d in g.nodes(data=True))
    flows = collections.Counter(d.get("flow", "?") for _, _, d in g.edges(data=True))
    payload = json.dumps(
        {
            "n": g.number_of_nodes(),
            "e": g.number_of_edges(),
            "t": sorted(texts.items()),
            "f": sorted((str(k), v) for k, v in flows.items()),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def graph_for(src: str, opt: str, timeout: float):
    """Compile src at one -O level -> ProgramML graph. None on failure."""
    try:
        G = pg.from_cpp(src, copts=[opt], language="c", version="10", timeout=timeout)
        return pg.to_networkx(G)
    except Exception:  # noqa: BLE001  (UnsupportedCompiler / GraphCreationError / timeout)
        return None


def partition_label(sigs: dict[str, str]) -> str:
    """Group -O levels by identical signature: '-O0 | -O1=-O2=-O3'."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for lvl in OPT_LEVELS:
        groups[sigs[lvl]].append(lvl)
    parts = ["=".join(v) for v in groups.values()]
    return " | ".join(sorted(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="ExeBench O-level distinctness gate")
    ap.add_argument("--n", type=int, default=300, help="programs to attempt")
    ap.add_argument("--split", default="test_real",
                    help="ExeBench split (test_real/valid_real/train_real_compilable/...)")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-compile timeout")
    ap.add_argument("--min-nodes", type=int, default=5,
                    help="skip programs whose -O0 graph is smaller (degenerate)")
    ap.add_argument("--out", default="probe_report.json")
    args = ap.parse_args()

    # size buckets (by -O0 node count) — O3's wins (vectorize/unroll/inline) only
    # show up on bigger code, so we stratify the distinctness numbers.
    BUCKETS = ((0, 30), (30, 100), (100, 300), (300, 10**9))

    def bucket_of(n: int) -> str:
        for lo, hi in BUCKETS:
            if lo <= n < hi:
                return f"{lo}-{hi if hi < 10**9 else 'inf'}"
        return "?"

    t0 = time.time()
    complete = 0          # compiled at all 4 levels (and >= min_nodes)
    attempted = 0
    compile_fail = 0      # at least one level failed
    degenerate = 0        # compiled but -O0 graph < min_nodes
    pair = {"O0!=O1": 0, "O1!=O2": 0, "O2!=O3": 0}
    strat = {bucket_of(lo): {"n": 0, "O1!=O2": 0, "O2!=O3": 0} for lo, _ in BUCKETS}
    partitions = collections.Counter()
    node_counts: list[int] = []
    examples = []

    for row in load_exebench_rows(args.n, args.split):
        attempted += 1
        if attempted == 1:
            print("PROBE::ROWKEYS " + json.dumps(sorted(row.keys()))[:500], flush=True)
        src = source_of(row)
        if src is None:
            continue
        graphs = {lvl: graph_for(src, lvl, args.timeout) for lvl in OPT_LEVELS}
        if any(g is None for g in graphs.values()):
            compile_fail += 1
            continue
        n0 = graphs["-O0"].number_of_nodes()
        if n0 < args.min_nodes:
            degenerate += 1
            continue
        sigs = {lvl: signature(g) for lvl, g in graphs.items()}
        complete += 1
        node_counts.append(n0)
        d01 = sigs["-O0"] != sigs["-O1"]
        d12 = sigs["-O1"] != sigs["-O2"]
        d23 = sigs["-O2"] != sigs["-O3"]
        pair["O0!=O1"] += d01
        pair["O1!=O2"] += d12
        pair["O2!=O3"] += d23
        b = strat[bucket_of(n0)]
        b["n"] += 1
        b["O1!=O2"] += d12
        b["O2!=O3"] += d23
        partitions[partition_label(sigs)] += 1
        if d23 and len(examples) < 6:  # capture rare O2!=O3 cases if any
            examples.append({lvl: g.number_of_nodes() for lvl, g in graphs.items()})
        if complete % 50 == 0:
            o2o3 = 100.0 * pair["O2!=O3"] / complete
            print(f"PROBE::PROGRESS complete={complete} O2!=O3={o2o3:.1f}%", flush=True)

    def pct(k: str) -> float:
        return 100.0 * pair[k] / complete if complete else 0.0

    strat_pct = {
        b: {
            "n": v["n"],
            "O1!=O2_pct": round(100.0 * v["O1!=O2"] / v["n"], 1) if v["n"] else None,
            "O2!=O3_pct": round(100.0 * v["O2!=O3"] / v["n"], 1) if v["n"] else None,
        }
        for b, v in strat.items()
    }
    report = {
        "split": args.split,
        "n_attempted": attempted,
        "n_complete": complete,
        "n_compile_fail": compile_fail,
        "n_degenerate": degenerate,
        "pair_distinct_pct": {k: round(pct(k), 1) for k in pair},
        "pair_distinct_count": dict(pair),
        "by_size": strat_pct,
        "partitions": dict(partitions.most_common()),
        "node_count_median": statistics.median(node_counts) if node_counts else None,
        "node_count_mean": round(statistics.mean(node_counts), 1) if node_counts else None,
        "node_count_max": max(node_counts) if node_counts else None,
        "examples_O2neqO3": examples,
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("PROBE::REPORT " + json.dumps(report, indent=2))
    o2o3 = pct("O2!=O3")
    verdict = "GREEN" if o2o3 >= 50 else ("AMBER" if o2o3 >= 15 else "RED")
    print(f"PROBE::GATE O2!=O3={o2o3:.1f}% verdict={verdict} (n_complete={complete})")
    print("PROBE::DONE")


if __name__ == "__main__":
    main()

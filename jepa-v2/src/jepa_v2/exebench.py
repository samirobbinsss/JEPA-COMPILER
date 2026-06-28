"""Read ExeBench directly from its HF tarballs (the loader script is broken with
modern `datasets`: it forces downloading every archive and can't stream).

Each split is one `<split>.tar.gz` holding `.jsonl.zst` shards; each json line is
`{"meta": {}, "text": {<record>}}`. The useful fields live under "text":
  func_def         - the C function definition
  real_deps        - real #include / typedef context (real splits)
  angha_deps       - synthesized context (synth splits)
  fname, path      - identity (used for deterministic train/val/test splits)
"""
from __future__ import annotations

import json
import tarfile
from typing import Iterator

SPLIT_TARBALL = {
    "test_real": "test_real.tar.gz",
    "valid_real": "valid_real.tar.gz",
    "test_synth": "test_synth.tar.gz",
    "valid_synth": "valid_synth.tar.gz",
    "train_real_compilable": "train_real_compilable.tar.gz",
    "train_synth_compilable": "train_synth_compilable.tar.gz",
    "train_real_simple_io": "train_real_simple_io.tar.gz",
}


def split_path(split: str) -> str:
    """Download (cached) the split tarball from the HF hub; return local path."""
    from huggingface_hub import hf_hub_download

    fname = SPLIT_TARBALL.get(split, split)
    return hf_hub_download("jordiae/exebench", fname, repo_type="dataset")


def iter_records(split: str, limit: int | None = None) -> Iterator[dict]:
    """Yield up to `limit` ExeBench `text` records (dicts) from a split."""
    import zstandard as zstd

    path = split_path(split)
    dctx = zstd.ZstdDecompressor()
    n = 0
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
                    n += 1
                    if limit is not None and n >= limit:
                        return


def record_name(rec: dict, idx: int) -> str:
    """Stable per-program name for deterministic splitting (see splits.py)."""
    path = rec.get("path") or ""
    fname = rec.get("fname") or "fn"
    return f"{path}::{fname}::{idx}" if path else f"{fname}::{idx}"


def source_of(rec: dict) -> str | None:
    """Build standalone-compilable C: deps + func_def. None if no func body."""
    func = rec.get("func_def")
    if not isinstance(func, str) or not func.strip():
        return None
    deps = rec.get("real_deps") or rec.get("angha_deps") or ""
    if not isinstance(deps, str):
        deps = ""
    return f"{deps}\n{func}\n"

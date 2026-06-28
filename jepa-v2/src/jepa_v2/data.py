"""Data pipeline: ProgramML graph -> PyG, vocab, and program-grouped batching.

The encoder (model.py) expects each graph as a PyG `Data` with:
  x               : LongTensor[N]  — node `text` vocab ids (0 = <unk>)
  edge_index_0/1/2: LongTensor[2,E]— one per ProgramML flow (control/data/call)

A "program" contributes 4 views (one per -O level). The loss groups views by
program (for z_sem) and by -O level (for z_speed), so the batch carries per-view
`prog` and `lvl` labels. ProgramData below teaches PyG how to offset the custom
multi-relation edge indices when several graphs are batched together.

ProgramML has no arm64 wheel — everything here runs on the pod. Import the dgl
stub BEFORE programl (programl_compat installs it).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from . import programl_compat  # noqa: F401  (side effect: dgl stub, before programl)
from .config import OPT_LEVELS

# ProgramML edge `flow` int -> relation index. Matches model.FLOW_TO_REL.
NUM_REL = 3
UNK = 0  # vocab id 0 reserved for unknown / padding


# --------------------------------------------------------------------------- #
# ProgramML -> lightweight intermediate (compile once, vocab later, tensors last)
# --------------------------------------------------------------------------- #
def compile_view(src: str, opt: str, timeout: float = 60.0):
    """Compile C `src` at one -O level via programl's bundled clang-10.

    Returns a networkx MultiDiGraph, or None on any failure.
    """
    import programl as pg  # imported here so the dgl stub is already installed

    try:
        G = pg.from_cpp(src, copts=[opt], language="c", version="10", timeout=timeout)
        return pg.to_networkx(G)
    except Exception:  # noqa: BLE001  (UnsupportedCompiler/GraphCreationError/timeout)
        return None


def extract(g) -> tuple[list[str], list[tuple[int, int, int]]]:
    """networkx ProgramML graph -> (node texts, edges as (u,v,rel)).

    Node order is fixed by list(g.nodes()); edges use that contiguous indexing.
    """
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    texts = [str(g.nodes[n].get("text", "<unk>")) for n in nodes]
    edges: list[tuple[int, int, int]] = []
    for u, v, d in g.edges(data=True):
        try:
            rel = int(d.get("flow", 0))
        except (TypeError, ValueError):
            rel = 0
        if 0 <= rel < NUM_REL:
            edges.append((idx[u], idx[v], rel))
    return texts, edges


# --------------------------------------------------------------------------- #
# Vocab over node `text`
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi

    @property
    def size(self) -> int:
        return max(self.stoi.values(), default=0) + 1

    @classmethod
    def build(cls, texts_iter, max_size: int) -> "Vocab":
        """Build from an iterable of token lists; top-(max_size-1) by frequency."""
        counter: collections.Counter = collections.Counter()
        for texts in texts_iter:
            counter.update(texts)
        stoi = {"<unk>": UNK}
        for tok, _ in counter.most_common(max_size - 1):
            stoi[tok] = len(stoi)
        return cls(stoi)

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([self.stoi.get(t, UNK) for t in texts], dtype=torch.long)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stoi))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# PyG Data that knows how to batch multi-relation edge indices
# --------------------------------------------------------------------------- #
class ProgramData(Data):
    """A single graph-view. edge_index_{r} are offset by num_nodes when batched."""

    def __inc__(self, key, value, *args, **kwargs):  # noqa: ANN001
        if key.startswith("edge_index_"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):  # noqa: ANN001
        if key.startswith("edge_index_"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def view_to_data(texts: list[str], edges: list[tuple[int, int, int]],
                 vocab: Vocab, lvl: int) -> ProgramData:
    """Build a ProgramData from extracted (texts, edges) using `vocab`."""
    x = vocab.encode(texts)
    n = x.size(0)
    per_rel: list[list[list[int]]] = [[[], []] for _ in range(NUM_REL)]
    for u, v, r in edges:
        per_rel[r][0].append(u)
        per_rel[r][1].append(v)
    data = ProgramData(x=x)
    data.num_nodes = n
    for r in range(NUM_REL):
        ei = torch.tensor(per_rel[r], dtype=torch.long) if per_rel[r][0] \
            else torch.empty(2, 0, dtype=torch.long)
        setattr(data, f"edge_index_{r}", ei)
    data.lvl = torch.tensor([lvl], dtype=torch.long)
    return data


# --------------------------------------------------------------------------- #
# Program-grouped dataset + collate
# --------------------------------------------------------------------------- #
class ProgramDataset(torch.utils.data.Dataset):
    """Each item is one program: a list of len(OPT_LEVELS) ProgramData views."""

    def __init__(self, programs: list[list[ProgramData]]):
        self.programs = programs

    def __len__(self) -> int:
        return len(self.programs)

    def __getitem__(self, i: int) -> list[ProgramData]:
        return self.programs[i]


def collate_programs(batch: list[list[ProgramData]]) -> Batch:
    """Flatten P programs x V views into a PyG Batch with `prog` and `lvl` labels.

    prog[i] = which program (0..P-1) view i came from (for z_sem invariance).
    lvl[i]  = which -O level (0..V-1)            (for z_speed invariance).
    """
    views: list[ProgramData] = []
    prog_ids: list[int] = []
    for p, prog in enumerate(batch):
        for view in prog:
            views.append(view)
            prog_ids.append(p)
    b = Batch.from_data_list(views)
    b.prog = torch.tensor(prog_ids, dtype=torch.long)
    # b.lvl is already concatenated from each view's .lvl -> shape [num_views]
    b.lvl = b.lvl.view(-1)
    return b


def make_loader(dataset: ProgramDataset, batch_programs: int, *,
                shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_programs, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_programs, drop_last=True,
    )


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #
def save_cache(programs: list[list[ProgramData]], names: list[str], path: str | Path) -> None:
    torch.save({"programs": programs, "names": names, "opt_levels": list(OPT_LEVELS)}, path)


def load_cache(path: str | Path) -> tuple[list[list[ProgramData]], list[str]]:
    blob = torch.load(path, weights_only=False)
    return blob["programs"], blob["names"]

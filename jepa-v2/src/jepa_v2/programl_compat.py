"""Make `import programl` work on the pod. IMPORT THIS BEFORE importing programl.

    import jepa_v2.programl_compat  # noqa: F401  (side effect: installs dgl stub)
    import programl as pg

Why: programl 0.3.2 does a top-level `from programl.transform_ops import to_dgl`,
which `import dgl`. dgl 2.1.0 on the pod fails to load its native graphbolt lib
(built for a different torch). We never use to_dgl — only to_networkx — so we
inject a fake `dgl` module that satisfies the import without loading native code.

The other programl fixes (protobuf<3.21, networkx<3.0, libtinfo5) are environment
pins handled by scripts/setup_pod.sh, not here.
"""
from __future__ import annotations

import sys
import types


def install_dgl_stub() -> None:
    if "dgl" in sys.modules and getattr(sys.modules["dgl"], "_jepa_stub", False):
        return
    dgl = types.ModuleType("dgl")
    dgl._jepa_stub = True
    hetero = types.ModuleType("dgl.heterograph")

    class DGLHeteroGraph:  # only needs to exist as an importable name
        pass

    hetero.DGLHeteroGraph = DGLHeteroGraph
    dgl.heterograph = hetero
    dgl.DGLHeteroGraph = DGLHeteroGraph
    sys.modules["dgl"] = dgl
    sys.modules["dgl.heterograph"] = hetero


install_dgl_stub()

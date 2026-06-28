#!/usr/bin/env bash
# Reproduce the full ProgramML environment on a fresh RunPod pod.
# Run this ONCE on the pod (it is idempotent). All 7 fixes are here; the rationale
# is in HANDOFF.md ("Why programl was painful").
#
# Usage (from your Mac, via the proxy):
#   scripts/pod.sh put scripts/setup_pod.sh
#   scripts/pod.sh run 'bash /workspace/jepa-v2/setup_pod.sh'
set -euo pipefail

ROOT=/workspace/jepa-v2
mkdir -p "$ROOT"
cd "$ROOT"

# 0. dedicated venv (the pod is SHARED — never touch system python)
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip

# 1-3. programl + all pinned deps (versions matter — see HANDOFF.md)
pip install -q \
  "programl==0.3.2" \
  "protobuf<3.21" \
  "torchdata==0.7.1" \
  "networkx<3.0" \
  packaging "setuptools<81" \
  pandas pyyaml pydantic psutil tqdm requests scipy \
  numpy

# torch / torch-geometric (pod already has torch 2.8 + CUDA; install PyG).
# NOTE: torch_geometric 2.8 is pure-python for our usage (GraphConv, pooling,
# utils.scatter all have torch fallbacks) — no torch-scatter/sparse build needed.
python3 -c "import torch" 2>/dev/null || pip install -q torch
pip install -q torch-geometric

# data + eval deps. datasets MUST be <3: ExeBench ships a loader SCRIPT and >=3.0
# removed script support. We bypass the script anyway (src/jepa_v2/exebench.py reads
# the tarballs directly) but keep the pin for parity.
pip install -q "datasets<3" zstandard huggingface_hub matplotlib scikit-learn

# 5. libtinfo.so.5 — programl's native clang2graph-10 needs the REAL ncurses-5
#    (a symlink to so.6 is rejected; it needs symbol NCURSES_TINFO_5.0.19991023).
if ! ldconfig -p | grep -q "libtinfo.so.5"; then
  cd /tmp
  wget -q http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb -O libtinfo5.deb
  apt-get install -y -qq ./libtinfo5.deb || dpkg -i libtinfo5.deb
  cd "$ROOT"
fi

# 4 + 6 + 7 are CODE-level (dgl stub, networkx pin already above, no to_pyg):
#   the dgl stub must be imported BEFORE `import programl` — it lives in
#   src/jepa_v2/programl_compat.py. Always `import jepa_v2.programl_compat` first.

echo "=== verifying programl ==="
python3 - <<'PYEOF'
import jepa_v2.programl_compat  # installs the dgl stub
import programl as pg
G = pg.from_cpp("int main(){int s=0;for(int i=0;i<10;i++)s+=i;return s;}")
g = pg.to_networkx(G)
print("OK programl works: nodes=%d edges=%d" % (g.number_of_nodes(), g.number_of_edges()))
PYEOF
echo "=== setup_pod.sh DONE ==="

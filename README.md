<h1 align="center">JEPA-v2 — Factored Program Embedding</h1>

<p align="center">
  <b>A self-supervised program encoder on <a href="https://github.com/ChrisCummins/ProGraML">ProgramML</a> graphs,
  whose embedding is split into two disentangled sub-spaces.</b>
</p>

---

## The idea

We learn a program embedding `z` that is **factored** into two concatenated blocks:

```
z = [  z_sem  |  z_speed  ]
```

| Block | Pulled together | Pushed apart | Captures |
|---|---|---|---|
| **`z_sem`** | the 4 `-O` levels of the **same source** | different sources | *what the code does* (invariant to optimization) |
| **`z_speed`** | the **same `-O` level** across different sources | different `-O` levels | *the optimization / "speed" profile* (invariant to the program) |

So for one source compiled at O0/O1/O2/O3: `z_sem` is ~identical across the four,
while `z_speed` separates them by level.

## Pipeline

```
ExeBench C  ──clang -O{0,1,2,3}──►  LLVM IR  ──ProgramML──►  graph (control+data+call)
   ──to_pyg──►  GNN encoder (trained from scratch)  ──►  z = [z_sem | z_speed]
```

- **Self-supervised, no manual labels.** The `-O` level is used only to *group*
  views (positives/negatives); it is never a classification target.
- **No masking** in v2 (unlike v1 / `../jepa-ir`): the learning signal is the
  factored invariance structure across `-O` levels and across programs.
- **Anti-collapse: VICReg** on each block separately, plus a **cross-covariance**
  term that forces `z_sem ⟂ z_speed` (the disentanglement).

## Why v2 (vs `../jepa-ir`)

1. **ProgramML** as the input representation — the standard GNN-on-IR graph,
   comparable to the literature, instead of the v1 hand-built graph.
2. v1 proved that on **AnghaBench** clang saturates at -O2 (O1≈O2≈O3 have an
   *identical* IR graph). v2 **gates on a probe** before training — and the probe
   (see Results) found the same on **ExeBench**: on isolated functions O2≈O3 too.
   So `z_speed` is honestly an *optimized-vs-not* axis, not a 4-way O-level split.

## Working on the shared pod

The B200 pod is **shared**. We isolate by environment (a dedicated venv under
`/workspace/jepa-v2`), and we never throttle the GPU. The RunPod SSH proxy is
quirky (no scp, no `%` in printf, commands via stdin) — use `scripts/pod.sh`:

```bash
scripts/pod.sh run 'python3 -c "import programl; print(1)"'
scripts/pod.sh put scripts/probe_exebench.py
scripts/pod.sh shell
```

## Results

Full pipeline (probe → cache → train → eval) runs end-to-end on the B200 pod.
Corpus: **ExeBench `train_real_compilable`**, 3k–8k functions, ProgramML graphs,
node-`text` vocab, 6-layer 3-relation GraphConv trunk, `sem_dim=96`/`speed_dim=32`.

### Step-1 gate — does the graph differ across `-O`? (`docs/results_gate_exebench.md`)

Probe of 555 ExeBench functions (programl's bundled **clang-10**):

| pair | % graphs distinct |
|---|---|
| **O0 ≠ O1** | **100 %** |
| O1 ≠ O2 | 25 % (46 % on functions ≥100 nodes) |
| **O2 ≠ O3** | **1.6 %** |

→ clang saturates at O2 on isolated functions (same as AnghaBench). The optimization
signal learnable from this corpus is essentially **one axis: O0 vs optimized**.

### Disentanglement (held-out test, 400 programs) (`docs/results_disentangle.md`)

Trained self-supervised, no labels (shipped encoder, corrected loss). Cosine
intra-vs-inter gap (higher = cleaner); off-target silhouette near 0 = disentangled:

| block | gap | off-target check (lower = better) |
|---|---|---|
| **z_sem** (program identity, `-O`-invariant) | **0.89** | silhouette by `-O` = −0.00 |
| **z_speed** (opt profile, program-invariant) | **0.52** | silhouette by program = −0.91 |

The factorization works: `z_sem` ignores the optimization level (and now spans ~72
of 96 dims — see below), `z_speed` ignores the program. `z_speed`'s moderate gap is
the honest ceiling: the gate showed the optimization signal is ~1-bit, so `z_speed`
is a single axis (O0 vs optimized). The figure below illustrates the factorization
(2D PCA; `z_sem` looks diffuse precisely because it is high-rank):

![PCA disentanglement](docs/figures/pca_disentangle.png)

### A loss bug, found adversarially, then fixed (`docs/loss_review.md`)

The latent first collapsed to **~3–5 of 128 dimensions**, and adding more data did
not help. The cause: the
group-invariance term **summed** over feature dims while VICReg's anti-collapse
terms **averaged**, hiding a factor of `D` that drowned out the rank-creating
covariance term. Fixing the normalization (`sum→mean`, covariance `÷D`) — same data,
same coefficients — gave a clean, **testable** confirmation:

| | before fix | after fix |
|---|---|---|
| **z_sem effective rank** (of 96) | **2.97** | **72.4** |
| z_sem dims for 90 % variance | 2 | 48 |
| z_sem gap | 0.77 | 0.90 |

i.e. the collapse was a loss bug, not a data limitation. (`z_speed` stays low-rank —
that one *is* the data: a ~1-bit signal, per the gate.) The fix is applied in the
shipped encoder; below, the corrected `z_sem` is diffuse in 2D because it spreads
over ~72 dimensions, while `z_speed` still resolves into the O0-vs-optimized axis:

![PCA, corrected loss (high-rank z_sem)](docs/figures/pca_highrank.png)

### Reproduce

```bash
scripts/pod.sh run 'python3 scripts/probe_exebench.py --n 800 --split test_real'
scripts/pod.sh run 'python3 scripts/build_cache.py --n 3000 --split train_real_compilable --out data/cache'
scripts/pod.sh run 'python3 scripts/train.py --cache data/cache --epochs 50'
scripts/pod.sh run 'python3 scripts/eval_disentangle.py --ckpt checkpoints/encoder.pt --cache data/cache'
```

Shipped encoder: `checkpoints/encoder.pt` (+ `checkpoints/vocab.json`) — trained on
8k ExeBench functions with the corrected loss.

## Repo layout

```
src/jepa_v2/   encoder (model.py), loss.py, vicreg.py, data.py, exebench.py, splits.py, config.py
scripts/       probe_exebench.py, build_cache.py, train.py, eval_disentangle.py, pod.sh, setup_pod.sh
docs/          results_gate_exebench.md, results_disentangle.md, loss_review.md, figures/
checkpoints/   encoder.pt + vocab.json (the deliverable)
HANDOFF.md     full context for picking the project up
```

## License

MIT — see [LICENSE](LICENSE).

# Adversarial review — `factored_loss` (why the latent collapses to ~3–5 dims)

> Multi-agent adversarial analysis, 2026-06-28. 5 attack lenses → independent
> verification (14/25 findings confirmed) → synthesis. The design is sound; the
> **implementation has two normalization bugs** that silently break VICReg's tuned
> 25/25/1 balance and bias the loss toward collapse.
>
> **STATUS: Fixes 1 & 2 are APPLIED and CONFIRMED.** Same data, same coefficients,
> only the normalization fixed → z_sem effective rank **2.97 → 72.4** (of 96),
> dims-for-90%-var 2 → 48, sem gap 0.77 → 0.90. The shipped `checkpoints/encoder.pt`
> uses the corrected loss. Fix 3 (speed head) is left as recommended future work —
> it changes `SPEED_GROUPS` to 2-class, deliberately kept 4-class for now.

## Executive summary

The objective (group-invariance + VICReg variance/covariance per block + cross-block
decorrelation) is correctly *structured* but mis-*normalized*. `_group_invariance`
**sums** the squared centroid distance over the D feature dims while every
anti-collapse term **averages** over D. That hides a factor of **D** in the
invariance weight: effective `sim ≈ 25·D ≈ 2400` (z_sem, D=96) / `800` (z_speed,
D=32) against `std=25` and a covariance term that is itself `(D−1)≈95×`
under-normalized. The only rank-creating force (covariance decorrelation) is
therefore ~`D(D−1)≈10⁴×` too weak — so the latent fills only as many dimensions as
there are trivially-separable groups and nothing spreads it further. This, not the
corpus size, is why raising 3000→8000 programs did nothing to the effective rank.

## Findings (ranked, deduped)

| # | Title | Sev | Mechanism | Fix |
|---|-------|-----|-----------|-----|
| 1 | Invariance **summed** over dims (effective `sim`×D) | **critical** | `loss.py` `_group_invariance` does `.sum(dim=1).mean()` vs VICReg mean-over-dims → effective sim = 25·D ≈ 2400 (sem)/800 (speed) vs std=25; attraction dominates anti-collapse by ~2 orders. | `((z-centroids[group])**2).mean()` |
| 2 | Covariance under-normalized by `(D−1)` | **critical** | `vicreg.py` `covariance_term` `.mean()` divides off-diag SS by `D(D−1)` not `D` → 95×(sem)/31×(speed) too weak; with #1 the decorr:inv ratio is off ~10⁴. | `off_diagonal(cov).pow(2).sum()/D` |
| 3 | Dimensional asymmetry of #1 | high | Same sum-vs-mean → the 96-dim sem block gets exactly 3× the collapse pressure of the 32-dim speed block. | resolved by #1 |
| 4 | No inter-group repulsion | high | `_group_invariance` only *pulls* to centroids; all spreading delegated to var+cov, which #1/#2 neutered → attraction runs unopposed into a low-rank blob. | add centroid repulsion (InfoNCE / cov-to-identity on centroids) after rebalancing |
| 5 | Cross-decorr weak, redundant, linear-only | medium | `cross_covariance_term` ÷ `Da·Db`=3072, weight 1; under the crossed prog×lvl batch it's ≈0 once invariance holds → duplicates L_sem/L_speed; enforces only linear decorrelation, not independence. | `cross.pow(2).sum()/max(Da,Db)`, raise weight, or HSIC; fix docstrings |
| 6 | `SPEED_GROUPS=(0,1,2,3)` over a ~1-bit signal | low | O1/O2/O3 IR byte-identical ~98% → 4-class label collapses to opt/no-opt; 32 dims for ~1 bit (data limit). | `SPEED_GROUPS=(0,1,1,1)`, shrink `speed_dim` |
| 7 | Identical O2/O3 views never deduped | low | `require_all_levels` emits 4 rows/program with O2==O3 → ~50% of each z_speed batch is one duplicated cluster, biasing var/cov + the z_sem centroid; ~25% wasted compute. | hash-dedup views / multiplicity weights |
| 8 | Non-detached centroid / docstring | low | centroid left attached; for a *mean* centroid the gradient is identical detached-or-not (Σ(zᵢ−c)=0) → docstring rationale is a misconception, not a bug. | correct docstring only |

## Top 3 fixes (priority order)

**Fix 1 — normalize invariance over feature dims** (`loss.py` `_group_invariance`) — dominant driver.
```python
# before:
return ((z - centroids[group]) ** 2).sum(dim=1).mean()
# after:
return ((z - centroids[group]) ** 2).mean()   # mean over BOTH dims and rows
```

**Fix 2 — restore VICReg covariance normalization** (`vicreg.py`).
```python
# covariance_term:
n, d = z.shape
z = z - z.mean(dim=0, keepdim=True)
cov = (z.T @ z) / max(n - 1, 1)
return off_diagonal(cov).pow(2).sum() / d                       # was .mean()
# cross_covariance_term:
cross = (a.T @ b) / max(n - 1, 1)
return cross.pow(2).sum() / max(a.size(1), b.size(1))           # was .mean()
```
After 1+2 the realized balance returns to ~25:25:1; likely raise `cov_coeff` above 1
given the low intrinsic dimensionality of the graph features.

**Fix 3 — match the speed head to the discriminable signal** (`config.py`).
```python
SPEED_GROUPS = (0, 1, 1, 1)   # opt vs no-opt: the only separable axis
speed_dim: int = 8            # ~1-bit signal does not need 32 dims
```
Pair with hash-dedup of identical O2/O3 views (finding 7).

## Does the loss implement the stated objective?

Partially — divergence is in normalization, not structure.
- **z_sem opt-invariance**: correctly coded and *over*-implemented (intra-program
  cosine → 1.000), but summing over D inflates attraction ~96× → dominates the
  regularizers that should preserve rank.
- **z_speed program-invariance**: correctly coded but supervised on a degenerate
  label (O1/O2/O3 IR identical) → ~1-bit target → single axis. Data/labeling, not
  loss-logic.
- **Decorrelation**: variance is fine; covariance is `(D−1)×` under-normalized and
  invariance `D×` over-weighted → realized balance ~`10⁴:25:1`, not the documented
  `25:25:1` → decorrelation is effectively off, no rank forms. The cross term is
  ≈0 by construction under the crossed sampler, so it adds no real disentanglement
  gradient (and zero linear cross-cov ≠ independence for two functions of the same
  pooled vector). The docstring claim that centroid-MSE "equals ½·MSE(view_a,view_b)"
  is the tell: VICReg's MSE is mean-over-dims, so the coded sum-over-dims form is
  `(D/2)·MSE_mean` — off by exactly the D that drives the collapse.
```

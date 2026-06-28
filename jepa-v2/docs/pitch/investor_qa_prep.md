# Investor Q&A prep — internal (do NOT put the raw version on a slide)

> Output of a 4-persona VC red-team (technical DD · market/GTM · moat · execution),
> 2026-06-28. Brutally honest on purpose. The deck's public Q&A slide carries the
> *confident* version of these rebuttals; this file is the founders' war-room.

## The single most honest sentence

> Everything sold — beating -O3, irrefutable demos, a data moat — sits downstream of
> an optimizer that is **0% built**; the only thing actually proven is an encoder
> that, on our own data, learned to tell whether clang already optimized — a
> distinction clang gives away for free.

Internalize this. Lead with the *vision*, but never claim the optimizer exists.

## The 6 objections you WILL get → honest rebuttal

1. **"You describe code, you don't optimize it."** (the whole search→codegen→verify→benchmark
   stack is unbuilt; z is a thermometer, not an engine.)
   → A disentangled `z_speed` is a genuine *prerequisite* for a learned cost model —
   necessary, not sufficient. ~95% of the thesis is still ahead of us; we won't pretend otherwise.

2. **"Your own gate kills the premise."** (O2≈O3 for 98% → the only signal is "is this O0";
   on this corpus, "compilers leave performance on the table" is empirically false.)
   → The O2 ceiling is plausibly an artifact of *isolated functions* (nothing to inline,
   loops too short to vectorize); headroom should return on whole programs — a hypothesis
   we will test next (cheap), not a proven claim.

3. **"Correctness is 100% of the risk and 0% addressed."**
   → Architecture is **propose-then-prove**: an independent verifier we can't fool bounds the
   model's aggressiveness. It isn't built yet, so today we cannot guarantee correctness.

4. **"The GTM is gated on a demo you can't produce, aimed at unpaid maintainers."**
   → Correct today. The honest interim is `z_speed` as *lead-scoring*, not proof; and we have
   not yet validated who actually pays (HFT/db/sim teams with perf budgets, not OSS volunteers).

5. **"No moat — ProGraML, ExeBench, VICReg, a 6-layer GNN are all public; incumbents
   (Meta LLM Compiler, Google MLGO, Nvidia) own the substrate + 1000× the data."**
   → True. The only durable moat — a proprietary corpus of *(graph → measured speedup on
   specific hardware)* pairs — does not exist yet. Building it is milestone #3.

6. **"'First JEPA on assembly' is mislabeled (it's VICReg multi-view, no masking), and the
   0.89 gap is partly tautological (O1/O2/O3 byte-identical 75–98%)."**
   → Accurately: self-supervised disentangled representation learning on IR graphs; the JEPA
   world-model is the aspiration. We should report the gap on *graph-distinct* pairs to
   separate learned semantics from free same-input invariance.

## The 3 milestones that make us fundable (the roadmap / the ask)

1. **First beat-O3 win.** One closed loop — encoder → search → codegen → verify → measured
   wall-clock — that beats `clang -O3` on a real hot loop on real hardware, correctness-verified.
   A single such datapoint collapses objections 1, 3, 4 at once.
2. **Headroom proof (days, not months).** Re-run the gate on whole-program corpora
   (SPEC / cBench / MiBench; O2 vs O3 vs Ofast/PGO/autovec) to show an exploitable runtime
   delta exists where the thesis lives. Go/no-go on the founding premise.
3. **Safety rail + moat seed.** A propose-then-verify loop on narrow kernels with zero
   correctness regressions, plus the first 10k compounding (graph → hardware-measured,
   correctness-checked speedup) records — and a head-to-head beating CompilerGym/MLGO.

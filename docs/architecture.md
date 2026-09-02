# Architecture

```text
crates/
  hodgeker/          library
    complex.rs       SC₂, grid, hole, orientations
    operators.rs     sparse B₁, B₂ → dense L₀, L₁, L₂
    spectra.rs       Hodge eigen-split of L_d, L_u, L₁
    projectors.rs    P_G, P_C, P_H and f = f_G+f_C+f_H
    kernel.rs        compositional / edge / graph Matérn
    gp.rs            Cholesky GP, Nyström, holdout split
    hodgelet.rs      spectral-wavelet energy features
    synth.rs         seeded grad / curl / mixed / ocean flows
    demo.rs          ocean-flow GP vs graph baseline
    io.rs            JSON, OFF, OBJ, signal CSV
    linag.rs         SymmetricEigen, Cholesky, pinv
  hodgeker-cli/      clap binary `hodgeker`
```

## Types

| Type | Meaning |
|------|---------|
| `VertexId` / `EdgeId` / `FaceId` | newtype indices |
| `EdgeSignal` | 1-cochain, `DVector<f64>` of length `N₁` |
| `SimplicialComplex2` | vertices + oriented edges + triangles |
| `HodgeOperators` | `sprs` incidence, dense Laplacians |
| `HodgeSpectra` | `(Λ_G, U_G)`, `(Λ_C, U_C)`, `U_H` |
| `HodgeMaternParams` | independent `(σ, κ, ν)` per Hodge block |

Reference orientation is increasing vertex label. That matches Yang et al.
and Lim; it is not a directed graph.

## Linear algebra choices

- Incidence is sparse (`sprs` triplets → CSR). For a 12×12 grid you have
  O(10²) edges; densifying `L₁` is the right trade.
- Eigensolves are `nalgebra::linalg::SymmetricEigen` (Jacobi), not ARPACK.
  Medium complexes (`N₁ ≲ 10³`) are the design point. If you need 10⁵
  edges, swap this module for a sparse solver — the kernel formulae do not
  change.
- GP regression is dense Cholesky on the **observed** block, with a
  1e-12…1e-3 jitter schedule. Nyström is optional and tested.
- The naive graph baseline is the **line-graph** Laplacian Matérn, not
  `Ψ(L_d)`. Curl lives in `ker L_d`; using that operator as a strawman
  lets a “graph” kernel cheat on vortices.

## Determinism

Synthetic flows and holdout splits use `ChaCha8Rng` with an explicit `u64`
seed. Grid search over a fixed hyperparameter lattice is deterministic.
Do not expect bit-identical Cholesky across BLAS backends; `nalgebra` here
is pure Rust.

## What is deliberately missing

- Weighted Hodge Laplacians (Grady–Polimeni / Schaub et al.). Easy to add;
  the tests would need a weighted chain identity.
- Full Hodgelet *classification* GPs (Alain et al.). We ship the feature
  map.
- PyO3. The library is usable from Python via JSON/CSV today; bindings are
  a follow-up, not a substitute for a correct kernel.

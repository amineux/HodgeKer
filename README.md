# HodgeKer

A graph Laplacian will happily smooth a vortex into a source. That is not a
bug in your optimiser. It is the operator.

**HodgeKer** is a Rust (edition 2021) library for **Hodge-compositional
Matérn kernels** on **simplicial 2-complexes** — vertices, oriented edges,
triangles. Graphs are the 1-skeleton, and they are not enough for flows.

This is original code. The mathematics is not. Read Yang, Borovitskiy &
Isufi (AISTATS 2024) before you cite this repo.

## Why graphs lie for flows

An edge flow is a 1-cochain. It has a divergence at vertices and a curl
around faces. Hodge theory splits it, uniquely and orthogonally, into

```text
f  =  f_grad   ⊕   f_curl   ⊕   f_harm
      curl-free    div-free     both (holes)
```

The graph Helmholtzian `L_d = B₁ᵀ B₁` is the wrong “graph-only” strawman
for curl: every swirl sits in its kernel. The honest naive baseline is a
**line-graph Matérn** — treat oriented flows as unsigned scalars on
adjacent edges, ignore faces. That kernel fights circulation. Hodge does
not.

The edge Hodge Laplacian

```text
L₁ = B₁ᵀ B₁ + B₂ B₂ᵀ = L_d + L_u
```

sees faces. A **compositional** kernel goes further and gives each Hodge
block its own Matérn hyperparameters (Yang et al., eq. 16):

```text
K = U_G Ψ_G(Λ_G) U_Gᵀ  +  U_C Ψ_C(Λ_C) U_Cᵀ  +  σ_H² U_H U_Hᵀ
```

That is the whole trick. Independent lengthscales for sources and swirls.
A harmonic channel for holes. Posterior uncertainty that knows which
component it is interpolating.

## What you get

- `SC₂` with typed `VertexId` / `EdgeId` / `FaceId` / `EdgeSignal`
- Sparse `B₁, B₂`; Hodge `L₀, L₁, L₂`; chain identity `B₁ B₂ = 0`
- Hodge projectors, idempotent and mutually orthogonal
- Compositional Matérn, non-HC edge Matérn, graph (down-only) Matérn
- Exact GP (Cholesky) + optional Nyström landmarks
- Hodgelet energy features (Alain et al. 2024/2025) — the map, not a fake
  classification leaderboard
- Seeded synthetic flows: gradient, curl, mixed, ocean-current cartoon
- CLI: build, decompose, fit, predict, dump CSV
- Tests that fail if the maths is wrong, including **Hodge GP vs line-graph
  Matérn on a curl-heavy flow**

## Build / test / run

```bash
cargo test --release --all
cargo run --release -p hodgeker-cli -- demo --nx 8 --ny 8 --seed 42 --out demo_out
```

The demo writes `demo_out/decomp.csv`, `pred.csv`, `metrics.csv`,
`metrics.json`. Plot midpoints yourself; this crate does not pretend to be
a visualisation stack.

```bash
# build a grid and a vortex
cargo run --release -p hodgeker-cli -- build --nx 8 --ny 8 \
    --out complex.json --signal flow.csv --kind ocean --seed 0

# Hodge split
cargo run --release -p hodgeker-cli -- decompose \
    --complex complex.json --signal flow.csv --out decomp_out

# GP on a holdout (Hodge / edge / graph kernels)
cargo run --release -p hodgeker-cli -- fit \
    --complex complex.json --signal flow.csv --kernel hodge --seed 0
```

JSON schema is `data/triangle.json`. OFF and OBJ (triangles / quads) are
accepted by `decompose` / `fit`.

## Measured on this repo (not a paper table)

Command, rustc 1.83.0, `cargo run --release -p hodgeker-cli`:

```text
hodgeker demo --nx 8 --ny 8 --seed 42 --holdout 0.4
```

Triangulated 8×8 vertex grid: **161 edges**, **98 faces**, Hodge dims
`(grad, curl, harm) = (63, 98, 0)`. Synthetic ocean-like flow is **99.8%
curl** (energy). 40% of edges held out. Hyperparameters fit by log-marginal
grid search, same lattice for every kernel.

| kernel | held-out MSE |
|--------|----------------|
| Hodge-compositional Matérn | **0.303** |
| line-graph Matérn (naive graph) | 0.558 |
| non-HC edge Matérn on `L₁` | 0.758 |

Line-graph / Hodge ratio **1.84**. The CI test is the same claim on a 6×6
grid, seed 7: Hodge **0.218** vs line-graph **0.405** (ratio **1.86**).

Those are this checkout’s numbers. They are not Yang et al.’s table, not
Alain et al.’s table, and not a rounded fantasy.

## Library sketch

```rust
use hodgeker::complex::SimplicialComplex2;
use hodgeker::kernel::{compositional_matern, HodgeMaternParams};
use hodgeker::operators::HodgeOperators;
use hodgeker::projectors::decompose;
use hodgeker::spectra::hodge_spectra;

let sc = SimplicialComplex2::grid(8, 8, true)?;
let ops = HodgeOperators::from_complex(&sc)?;
assert!(ops.chain_identity_residual() < 1e-10);

let sp = hodge_spectra(&ops)?;
let parts = decompose(&sp, &flow);
let k = compositional_matern(&sp, &HodgeMaternParams::default());
```

See [`docs/math.md`](docs/math.md) for the SPDE, and
[`docs/architecture.md`](docs/architecture.md) for crate layout.

## Citations

HodgeKer implements published constructions. It does not author them.

1. **Yang, Borovitskiy, Isufi.** Hodge-Compositional Edge Gaussian
   Processes. AISTATS 2024.
   https://proceedings.mlr.press/v238/yang24e/yang24e.pdf
2. **Alain, Takao, Rieck, Dong, Noutahi.** Graph Classification GPs via
   Hodgelet Spectral Features. arXiv:2410.10546 (NeurIPS 2024 workshop
   spotlight).
3. **Alain, Takao, Rieck, Dong, Noutahi.** Graph and Simplicial Complex
   Prediction GP via Hodgelet Representations. arXiv:2505.10877.
4. **Alain, Takao, Paige, Deisenroth.** Gaussian Processes on Cellular
   Complexes. arXiv:2311.01198.
5. **Lim.** Hodge Laplacians on Graphs. *SIAM Review*, 2020.

Related: Borovitskiy et al., Matérn GPs on graphs (AISTATS 2021);
Rasmussen & Williams (2006); Crane, Discrete Differential Geometry.

## License

MIT. See [`LICENSE`](LICENSE).

# Hodge, Matérn, and why the graph Laplacian is the wrong operator for flows

This note is a reading guide, not a paper. The mathematics is due to the
authors cited below. HodgeKer is an independent implementation.

## Discrete exterior calculus on an `SC₂`

A simplicial 2-complex is a triple `(V, E, T)`: vertices, oriented edges, and
triangular faces, downward-closed. Graphs are the 1-skeleton (`T = ∅`).

Functions on these cells are cochains:

| degree | lives on | discrete name |
|--------|----------|----------------|
| 0      | vertices | scalar / potential `f₀` |
| 1      | edges    | flow `f₁` (alternating: reversing orientation flips the sign) |
| 2      | triangles| face circulation `f₂` |

The coboundary / boundary pair is encoded by incidence matrices. We follow
Yang, Borovitskiy & Isufi (AISTATS 2024), which itself follows Lim (SIAM Review
2020):

- `B₁ ∈ ℝ^{N₀×N₁}`: for an oriented edge `e = [i, j]`,
  `[B₁]_{i e} = −1`, `[B₁]_{j e} = +1`.
- `B₂ ∈ ℝ^{N₁×N₂}`: for an oriented triangle `t = [i, j, k]`,
  `+1` on `[i, j]` and `[j, k]`, `−1` on `[i, k]`.

Then

```text
grad  f₀ = B₁ᵀ f₀     (node potential → edge flow)
div   f₁ = B₁  f₁     (net outflow at vertices)
curl  f₁ = B₂ᵀ f₁     (circulation around faces)
```

The chain identity `curl ∘ grad = 0` is the matrix identity `B₂ᵀ B₁ᵀ = 0`, or
equivalently `B₁ B₂ = 0`. HodgeKer tests this on every mesh it ships.

## Hodge Laplacians

```text
L₀ = B₁ B₁ᵀ                         graph Laplacian on vertices
L₁ = B₁ᵀ B₁ + B₂ B₂ᵀ = L_d + L_u    edge Hodge / Helmholtzian
L₂ = B₂ᵀ B₂                         triangle Laplacian
```

`L_d` sees adjacency of edges *through vertices*. `L_u` sees adjacency of
edges *through faces*. Drop the faces and you drop `L_u`. That is the graph
lie: a 1-skeleton cannot tell a swirl from a source once both have been
projected onto vertices, and on edges it treats every divergence-free mode
as a kernel vector of `L_d`.

## Hodge decomposition (edges)

The Hodge theorem on a 2-complex splits the edge space orthogonally

```text
ℝ^{N₁} = im(B₁ᵀ)  ⊕  im(B₂)  ⊕  ker(L₁)
         gradient     curl       harmonic
         (curl-free)  (div-free) (div- and curl-free)
```

In the eigenbasis of `L₁` this is Yang et al., eqs. (12)–(13):

- `U_G`: eigenvectors of `L_d` with `λ > 0`
- `U_C`: eigenvectors of `L_u` with `λ > 0`
- `U_H`: eigenvectors of `L₁` with `λ = 0`  (`dim ker L₁ = β₁`)

HodgeKer computes these three blocks with a dense self-adjoint eigensolve
(`nalgebra::SymmetricEigen`) after assembling sparse `B₁, B₂`. Projectors
are `P_□ = U_□ U_□ᵀ`. They are idempotent and mutually orthogonal; the test
suite checks both.

A harmonic edge flow circulates around 1-dimensional holes. Fill every
triangle and you kill `β₁` on a disk. Punch a hole (omit faces) and a
harmonic mode appears. That is homology, not a regulariser trick.

## Matérn kernels from SPDEs

On graphs, Borovitskiy et al. (2021) obtained Matérn GPs as solutions of

```text
(2ν/κ² I + L₀)^{ν/2} f₀ = w₀,    w₀ ~ 𝒩(0, I)
```

which yields the covariance `(2ν/κ² I + L₀)^{-ν}`. Yang et al. (2024) put
the same SPDE on **edges**, with `L₁` in place of `L₀`, and then — this is
the compositional step — **split the operator across Hodge subspaces**.

For `□ ∈ {G, C}`:

```text
Ψ_□(Λ_□) = σ_□² ( 2ν_□/κ_□² I + Λ_□ )^{-ν_□}
K_□      = U_□ Ψ_□(Λ_□) U_□ᵀ
```

The harmonic block is a scaled projector, `K_H = σ_H² U_H U_Hᵀ`, because
`Λ_H = 0` (Yang et al., remark after eq. 16). The Hodge-compositional kernel
is the sum of three independent GPs:

```text
K = K_G + K_C + K_H
```

Hyperparameters are *not* shared. A curl-dominated ocean current can take a
long curl lengthscale and a near-zero gradient variance; a graph Matérn
cannot make that choice, because on `L_d` the entire curl space is the
kernel and receives a single scalar variance.

The non-compositional “edge Matérn” `Ψ(L₁)` (Yang et al., eq. 6) is also
implemented. It sees faces, but it ties the three Hodge blocks to one
`(σ, κ, ν)`.

## Graphs lie for flows

Three concrete failure modes of a graph-only kernel (`Ψ(L_d)` or a
line-graph Matérn):

1. **Blindness to orientation.** The line-graph Laplacian treats two
   edges that share a vertex as “nearby scalars.” A circulating flow has
   opposite signs on those edges by construction; the graph kernel fights
   the vortex. Hodge `L_u` *is* that circulation.
2. **Wrong smoothness on `L_d`?** Careful: `im(B₂) ⊂ ker(L_d)`, so a
   Matérn kernel on the down Helmholtzian secretly puts a white prior on
   the entire curl space. With enough observations that can interpolate a
   vortex — not because graphs understand flow, but because they collapsed
   every swirl into one kernel. HodgeKer’s demo therefore compares against
   the **line-graph** Matérn, which is the kernel you actually get if you
   pretend an edge flow is a scalar graph signal.
3. **No harmonic channel.** Holes are `ker(L₁)`, not a line-graph leftover.

HodgeKer’s demo builds a triangulated grid, places a smooth vortex on the
faces (`f = B₂ ψ` plus a weak eastward drift), holds out ~40% of edges, and
fits both kernels by log-marginal grid search. The number that belongs in
the README is the number that command prints — not a table copied from a
paper.

## Hodgelets

Alain et al. (NeurIPS 2024 workshop; 2025 follow-up) turn Hodge spectral
filters into Euclidean features for *graph/complex-level* GPs. HodgeKer
implements the feature map, not their full classification pipeline:

```text
ψ(s, λ) = (s λ) exp(−(s λ)²)     (Hammond-style band-pass)
a(λ)    = exp(−γ λ)              (low-pass)
A_□(s)  = ‖ U_□ ψ(s Λ_□) U_□ᵀ f ‖₂
```

Concatenate `(A_G, A_C, A_H)` across scales. The signature is invariant to
simplex ordering. Feed it to whatever Euclidean kernel you like; that part
is deliberately out of scope.

## Gaussian process regression

Exact GP on observed edges `I` (Rasmussen & Williams):

```text
μ_* = K_{*I} (K_{II} + σ_ε² I)^{-1} y
v_* = k_{**} − k_{*I} (K_{II} + σ_ε² I)^{-1} k_{I*}
```

Cholesky with a jitter schedule. For larger `N₁`, Nyström with strided
landmarks: `K ≈ C W⁺ Cᵀ`. Spectral truncation (drop high eigenvalues inside
each Hodge block) is the other obvious inducing scheme; the code path is
the Nyström one because it applies uniformly to Hodge, edge, and graph
kernels.

## Citations (read these; do not attribute them to this repo)

1. Maosheng Yang, Viacheslav Borovitskiy, Elvin Isufi.
   *Hodge-Compositional Edge Gaussian Processes.*
   AISTATS 2024. https://proceedings.mlr.press/v238/yang24e/yang24e.pdf
   arXiv:2310.19450.
2. Mathieu Alain, So Takao, Bastian Rieck, Xiaowen Dong, Emmanuel Noutahi.
   *Graph Classification Gaussian Processes via Hodgelet Spectral Features.*
   NeurIPS 2024 workshop spotlight. https://arxiv.org/abs/2410.10546
3. Mathieu Alain, So Takao, Bastian Rieck, Xiaowen Dong, Emmanuel Noutahi.
   *Graph and Simplicial Complex Prediction Gaussian Process via the Hodgelet
   Representations.* arXiv:2505.10877.
4. Mathieu Alain, So Takao, Brooks Paige, Marc Deisenroth.
   *Gaussian Processes on Cellular Complexes.* ICML 2024. arXiv:2311.01198.
5. Lek-Heng Lim. *Hodge Laplacians on Graphs.* SIAM Review 62(3), 2020.
6. Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, Marc Deisenroth.
   *Matérn Gaussian Processes on Graphs.* AISTATS 2021.
7. Carl E. Rasmussen, Christopher K. I. Williams.
   *Gaussian Processes for Machine Learning.* MIT Press, 2006.
8. Keenan Crane. *Discrete Differential Geometry: An Applied Introduction.*

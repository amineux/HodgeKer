//! Matérn spectral kernels on Hodge eigenspaces.
//!
//! Yang, Borovitskiy & Isufi, AISTATS 2024, eq. (16):
//!
//! ```text
//! Ψ_□(Λ_□) = σ_□² ( 2ν_□ / κ_□²  I  +  Λ_□ )^{-ν_□}
//! K_□      = U_□ Ψ_□(Λ_□) U_□ᵀ
//! ```
//!
//! for `□ ∈ {G, C}`. The harmonic kernel is the scaled projector
//! `K_H = σ_H² U_H U_Hᵀ` (their remark below eq. 16: `Ψ_H(0) = σ_H²`).
//!
//! The Hodge-compositional kernel is the sum of three independent kernels.
//!
//! The naive **graph** baseline is a Matérn kernel on the *line graph*
//! (edges of the complex become vertices; adjacency = shared original vertex).
//! That kernel treats oriented flows as unsigned scalars and has no access
//! to faces, so it fights circulation. (Matérn on `L_d = B₁ᵀ B₁` is a
//! different, stronger operator: curl sits in `ker L_d`, so it is *not* a
//! fair "graphs ignore triangles" strawman.)

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::complex::SimplicialComplex2;
use crate::error::{HodgekerError, Result};
use crate::spectra::HodgeSpectra;

/// Scalar Matérn hyperparameters on one Hodge (or graph) component.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaternParams {
    /// Variance `σ² > 0`.
    pub variance: f64,
    /// Spatial scale `κ > 0` in the SPDE `(2ν/κ² I + L)^{ν/2} f = W`.
    pub kappa: f64,
    /// Smoothness `ν > 0`.
    pub nu: f64,
}

impl MaternParams {
    /// Default-ish Matérn-3/2.
    pub fn matern32(variance: f64, kappa: f64) -> Self {
        Self {
            variance,
            kappa,
            nu: 1.5,
        }
    }

    /// Spectral multiplier `σ² (2ν/κ² + λ)^{-ν}`.
    pub fn psi(&self, lambda: f64) -> f64 {
        let shift = 2.0 * self.nu / (self.kappa * self.kappa);
        self.variance * (shift + lambda).powf(-self.nu)
    }
}

impl Default for MaternParams {
    fn default() -> Self {
        Self::matern32(1.0, 1.0)
    }
}

/// Independent Matérn hyperparameters for the three Hodge components.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HodgeMaternParams {
    /// Curl-free / gradient component.
    pub grad: MaternParams,
    /// Div-free / curl component.
    pub curl: MaternParams,
    /// Harmonic variance `σ_H²` (lengthscale unused).
    pub harm_variance: f64,
    /// Observation noise `σ_ε²`.
    pub noise: f64,
}

impl Default for HodgeMaternParams {
    fn default() -> Self {
        Self {
            grad: MaternParams::matern32(1.0, 1.0),
            curl: MaternParams::matern32(1.0, 1.0),
            harm_variance: 1.0,
            noise: 1e-3,
        }
    }
}

/// Which edge kernel to assemble.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelKind {
    /// Hodge-compositional Matérn (`K_G + K_C + K_H`).
    HodgeMatern,
    /// Non-compositional edge Matérn on the full `L₁` (shared hypers).
    EdgeMatern,
    /// Matérn on the line-graph Laplacian (naive graph kernel on edges).
    GraphMatern,
}

/// Hodge-compositional kernel matrix `K = K_G + K_C + K_H`.
pub fn compositional_matern(sp: &HodgeSpectra, p: &HodgeMaternParams) -> DMatrix<f64> {
    let n = sp.n_edges();
    let mut k = DMatrix::zeros(n, n);
    k += spectral_matern(&sp.grad_evecs, &sp.grad_evals, &p.grad);
    k += spectral_matern(&sp.curl_evecs, &sp.curl_evals, &p.curl);
    k += p.harm_variance * projector(&sp.harm_evecs);
    k
}

/// Non-HC edge Matérn `K = Ψ(L₁)` with one set of hyperparameters.
pub fn edge_matern(sp: &HodgeSpectra, p: &MaternParams) -> DMatrix<f64> {
    spectral_matern(&sp.l1_evecs, &sp.l1_evals, p)
}

/// Spectral Matérn on a provided eigenbasis (line graph, `L_d`, …).
pub fn graph_matern(evals: &DVector<f64>, evecs: &DMatrix<f64>, p: &MaternParams) -> DMatrix<f64> {
    spectral_matern(evecs, evals, p)
}

/// Assemble a kernel by kind. For [`KernelKind::GraphMatern`], pass the
/// line-graph eigenpairs from [`line_graph_spectrum`].
pub fn assemble(
    kind: KernelKind,
    sp: &HodgeSpectra,
    hodge: &HodgeMaternParams,
    shared: &MaternParams,
    graph_evals: &DVector<f64>,
    graph_evecs: &DMatrix<f64>,
) -> Result<DMatrix<f64>> {
    match kind {
        KernelKind::HodgeMatern => Ok(compositional_matern(sp, hodge)),
        KernelKind::EdgeMatern => Ok(edge_matern(sp, shared)),
        KernelKind::GraphMatern => {
            if graph_evecs.nrows() != sp.n_edges() {
                return Err(HodgekerError::Dimension(
                    "graph kernel: eigenvectors do not match N₁".into(),
                ));
            }
            Ok(graph_matern(graph_evals, graph_evecs, shared))
        }
    }
}

/// Rank-1 sum `U diag(ψ(λ)) Uᵀ`.
pub fn spectral_matern(
    evecs: &DMatrix<f64>,
    evals: &DVector<f64>,
    p: &MaternParams,
) -> DMatrix<f64> {
    let n = evecs.nrows();
    let m = evecs.ncols();
    if m == 0 {
        return DMatrix::zeros(n, n);
    }
    let mut scaled = evecs.clone();
    for k in 0..m {
        let amp = p.psi(evals[k]).max(0.0).sqrt();
        for r in 0..n {
            scaled[(r, k)] *= amp;
        }
    }
    &scaled * scaled.transpose()
}

fn projector(u: &DMatrix<f64>) -> DMatrix<f64> {
    let n = u.nrows();
    if u.ncols() == 0 {
        DMatrix::zeros(n, n)
    } else {
        u * u.transpose()
    }
}

/// Full eigenpairs of `L_d` (including zeros).
pub fn down_spectrum(l_down: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    crate::linag::sym_eig(l_down)
}

/// Unnormalized Laplacian of the line graph of the 1-skeleton.
///
/// Line-graph vertices are the complex's edges; two are adjacent when they
/// share a vertex. Faces and orientation are ignored — this is the "treat a
/// flow as a scalar graph signal on edges" baseline.
pub fn line_graph_laplacian(sc: &SimplicialComplex2) -> DMatrix<f64> {
    let n = sc.n_edges();
    let mut adj: DMatrix<f64> = DMatrix::zeros(n, n);
    let mut incident = vec![Vec::new(); sc.n_vertices()];
    for (e, edge) in sc.edges().iter().enumerate() {
        incident[edge.src.index()].push(e);
        incident[edge.dst.index()].push(e);
    }
    for star in &incident {
        for i in 0..star.len() {
            for j in (i + 1)..star.len() {
                let a = star[i];
                let b = star[j];
                adj[(a, b)] += 1.0;
                adj[(b, a)] += 1.0;
            }
        }
    }
    let mut lap = DMatrix::zeros(n, n);
    for i in 0..n {
        let mut deg = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            deg += adj[(i, j)];
            lap[(i, j)] = -adj[(i, j)];
        }
        lap[(i, i)] = deg;
    }
    lap
}

/// Eigenpairs of [`line_graph_laplacian`].
pub fn line_graph_spectrum(sc: &SimplicialComplex2) -> (DVector<f64>, DMatrix<f64>) {
    crate::linag::sym_eig(&line_graph_laplacian(sc))
}

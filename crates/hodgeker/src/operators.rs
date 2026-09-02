//! Boundary operators `B₁, B₂` and Hodge Laplacians `L₀, L₁, L₂`.
//!
//! Incidence conventions follow Yang, Borovitskiy & Isufi, AISTATS 2024, §2:
//!
//! * `B₁ ∈ R^{N₀ × N₁}`: `[B₁]_{i e} = -1`, `[B₁]_{j e} = +1` for `e = [i, j]`.
//! * `B₂ ∈ R^{N₁ × N₂}`: for `t = [i, j, k]`, `+1` on `[i, j]` and `[j, k]`,
//!   `-1` on `[i, k]`.
//! * `L₀ = B₁ B₁ᵀ` (graph Laplacian)
//! * `L₁ = B₁ᵀ B₁ + B₂ B₂ᵀ = L_d + L_u` (edge Hodge / Helmholtzian)
//! * `L₂ = B₂ᵀ B₂`
//!
//! The chain identity `B₁ B₂ = 0` (equivalently `curl ∘ grad = 0`) is tested
//! in the integration suite.

use nalgebra::DMatrix;
use sprs::{CsMat, TriMat};

use crate::complex::SimplicialComplex2;
use crate::error::Result;
use crate::linag::sprs_to_dense;

/// Sparse incidence plus dense Hodge Laplacians (dense eigen is the intended
/// path for medium complexes; incidence stays sparse).
#[derive(Clone, Debug)]
pub struct HodgeOperators {
    /// `B₁` (`N₀ × N₁`).
    pub b1: CsMat<f64>,
    /// `B₂` (`N₁ × N₂`).
    pub b2: CsMat<f64>,
    /// Graph Laplacian `L₀`.
    pub l0: DMatrix<f64>,
    /// Down Hodge `L_d = B₁ᵀ B₁`.
    pub l1_down: DMatrix<f64>,
    /// Up Hodge `L_u = B₂ B₂ᵀ`.
    pub l1_up: DMatrix<f64>,
    /// Edge Hodge Laplacian `L₁`.
    pub l1: DMatrix<f64>,
    /// Triangle Hodge `L₂`.
    pub l2: DMatrix<f64>,
}

impl HodgeOperators {
    /// Assemble operators from a 2-complex.
    pub fn from_complex(sc: &SimplicialComplex2) -> Result<Self> {
        let n0 = sc.n_vertices();
        let n1 = sc.n_edges();
        let n2 = sc.n_faces();

        let mut b1_tri = TriMat::new((n0, n1));
        for (e, edge) in sc.edges().iter().enumerate() {
            b1_tri.add_triplet(edge.src.index(), e, -1.0);
            b1_tri.add_triplet(edge.dst.index(), e, 1.0);
        }
        let b1: CsMat<f64> = b1_tri.to_csr();

        let mut b2_tri = TriMat::new((n1, n2));
        for (t, face) in sc.faces().iter().enumerate() {
            let [i, j, k] = face.verts;
            // t = [i, j, k] with i < j < k. Edges: [i,j] +1, [j,k] +1, [i,k] -1.
            let e_ij = sc.edge_id(i, j)?;
            let e_jk = sc.edge_id(j, k)?;
            let e_ik = sc.edge_id(i, k)?;
            b2_tri.add_triplet(e_ij.index(), t, sc.edge_sign(i, j)?);
            b2_tri.add_triplet(e_jk.index(), t, sc.edge_sign(j, k)?);
            b2_tri.add_triplet(e_ik.index(), t, -sc.edge_sign(i, k)?);
        }
        let b2: CsMat<f64> = b2_tri.to_csr();

        let b1d = sprs_to_dense(&b1);
        let b2d = if n2 == 0 {
            DMatrix::zeros(n1, 0)
        } else {
            sprs_to_dense(&b2)
        };

        let l0 = &b1d * b1d.transpose();
        let l1_down = b1d.transpose() * &b1d;
        let l1_up = if n2 == 0 {
            DMatrix::zeros(n1, n1)
        } else {
            &b2d * b2d.transpose()
        };
        let l1 = &l1_down + &l1_up;
        let l2 = if n2 == 0 {
            DMatrix::zeros(0, 0)
        } else {
            b2d.transpose() * &b2d
        };

        Ok(Self {
            b1,
            b2,
            l0,
            l1_down,
            l1_up,
            l1,
            l2,
        })
    }

    /// Dense `B₁`.
    pub fn b1_dense(&self) -> DMatrix<f64> {
        sprs_to_dense(&self.b1)
    }

    /// Dense `B₂`.
    pub fn b2_dense(&self) -> DMatrix<f64> {
        if self.b2.cols() == 0 {
            DMatrix::zeros(self.b2.rows(), 0)
        } else {
            sprs_to_dense(&self.b2)
        }
    }

    /// Discrete gradient: `grad f₀ = B₁ᵀ f₀` (node potential → edge flow).
    pub fn grad(&self, f0: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        self.b1_dense().transpose() * f0
    }

    /// Discrete divergence: `div f₁ = B₁ f₁`.
    pub fn div(&self, f1: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        &self.b1_dense() * f1
    }

    /// Discrete curl: `curl f₁ = B₂ᵀ f₁` (circulation around faces).
    pub fn curl(&self, f1: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        if self.b2.cols() == 0 {
            return nalgebra::DVector::zeros(0);
        }
        self.b2_dense().transpose() * f1
    }

    /// `||B₁ B₂||_F` — must be ~0.
    pub fn chain_identity_residual(&self) -> f64 {
        if self.b2.cols() == 0 {
            return 0.0;
        }
        let prod = &self.b1_dense() * self.b2_dense();
        crate::linag::frob(&prod)
    }
}

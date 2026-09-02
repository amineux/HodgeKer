//! Hodge eigenspaces of `L₁`: gradient ⊕ curl ⊕ harmonic.

use nalgebra::{DMatrix, DVector};

use crate::error::Result;
use crate::linag::{spectral_tol, sym_eig, take_columns};
use crate::operators::HodgeOperators;

/// Spectral Hodge splitting of the edge space.
///
/// Following Yang et al. (AISTATS 2024, eq. 12–13):
///
/// * `U_G` — eigenvectors of `L_d = B₁ᵀ B₁` with `λ > 0`  (`im B₁ᵀ`)
/// * `U_C` — eigenvectors of `L_u = B₂ B₂ᵀ` with `λ > 0`  (`im B₂`)
/// * `U_H` — eigenvectors of `L₁` with `λ ≈ 0`            (`ker L₁`)
#[derive(Clone, Debug)]
pub struct HodgeSpectra {
    /// Nonzero eigenvalues of `L_d`.
    pub grad_evals: DVector<f64>,
    /// Gradient eigenvectors (`N₁ × n_grad`).
    pub grad_evecs: DMatrix<f64>,
    /// Nonzero eigenvalues of `L_u`.
    pub curl_evals: DVector<f64>,
    /// Curl eigenvectors (`N₁ × n_curl`).
    pub curl_evecs: DMatrix<f64>,
    /// Harmonic eigenvectors (`N₁ × n_harm`), `Λ_H = 0`.
    pub harm_evecs: DMatrix<f64>,
    /// Full `L₁` eigenvalues (ascending).
    pub l1_evals: DVector<f64>,
    /// Full `L₁` eigenvectors (columns).
    pub l1_evecs: DMatrix<f64>,
    /// Relative cutoff used to split kernel / range.
    pub tol: f64,
}

impl HodgeSpectra {
    /// `N₁`.
    pub fn n_edges(&self) -> usize {
        self.l1_evecs.nrows()
    }

    /// `dim im(B₁ᵀ)`.
    pub fn n_grad(&self) -> usize {
        self.grad_evecs.ncols()
    }

    /// `dim im(B₂)`.
    pub fn n_curl(&self) -> usize {
        self.curl_evecs.ncols()
    }

    /// `dim ker(L₁) = β₁` (for a connected 2-complex, the first Betti number).
    pub fn n_harm(&self) -> usize {
        self.harm_evecs.ncols()
    }

    /// Betti-style report `(n_grad, n_curl, n_harm)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.n_grad(), self.n_curl(), self.n_harm())
    }
}

/// Compute the Hodge spectral splitting of `ops.l1`.
pub fn hodge_spectra(ops: &HodgeOperators) -> Result<HodgeSpectra> {
    let (ld_evals, ld_evecs) = sym_eig(&ops.l1_down);
    let (lu_evals, lu_evecs) = sym_eig(&ops.l1_up);
    let (l1_evals, l1_evecs) = sym_eig(&ops.l1);

    let tol_d = spectral_tol(&ld_evals, 1e-8);
    let tol_u = spectral_tol(&lu_evals, 1e-8);
    let tol_1 = spectral_tol(&l1_evals, 1e-8);
    let tol = tol_d.max(tol_u).max(tol_1);

    let (grad_evals, grad_evecs) = take_columns(&ld_evals, &ld_evecs, |l| l > tol);
    let (curl_evals, curl_evecs) = take_columns(&lu_evals, &lu_evecs, |l| l > tol);
    let (_, harm_evecs) = take_columns(&l1_evals, &l1_evecs, |l| l.abs() <= tol);

    Ok(HodgeSpectra {
        grad_evals,
        grad_evecs,
        curl_evals,
        curl_evecs,
        harm_evecs,
        l1_evals,
        l1_evecs,
        tol,
    })
}

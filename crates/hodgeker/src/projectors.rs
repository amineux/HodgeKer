//! Orthogonal Hodge projectors and the decomposition of an edge signal.
//!
//! `P_□ = U_□ U_□ᵀ` for `□ ∈ {G, C, H}`. These are orthogonal projectors onto
//! `im B₁ᵀ`, `im B₂`, and `ker L₁` respectively (Hodge theorem).

use nalgebra::{DMatrix, DVector};

use crate::ids::EdgeSignal;
use crate::linag::frob;
use crate::spectra::HodgeSpectra;

/// `P_G, P_C, P_H` as dense `N₁ × N₁` matrices.
#[derive(Clone, Debug)]
pub struct HodgeProjectors {
    /// Projector onto gradient (curl-free) space.
    pub grad: DMatrix<f64>,
    /// Projector onto curl (div-free) space.
    pub curl: DMatrix<f64>,
    /// Projector onto harmonic space.
    pub harm: DMatrix<f64>,
}

impl HodgeProjectors {
    /// Build `P = U Uᵀ` from a Hodge spectrum.
    pub fn from_spectra(sp: &HodgeSpectra) -> Self {
        Self {
            grad: projector(&sp.grad_evecs),
            curl: projector(&sp.curl_evecs),
            harm: projector(&sp.harm_evecs),
        }
    }

    /// Idempotence residuals `||P² − P||_F`.
    pub fn idempotence_residuals(&self) -> (f64, f64, f64) {
        (
            frob(&(&self.grad * &self.grad - &self.grad)),
            frob(&(&self.curl * &self.curl - &self.curl)),
            frob(&(&self.harm * &self.harm - &self.harm)),
        )
    }

    /// Mutual products; Hodge orthogonality wants these near zero.
    pub fn orthogonality_residuals(&self) -> (f64, f64, f64) {
        (
            frob(&(&self.grad * &self.curl)),
            frob(&(&self.grad * &self.harm)),
            frob(&(&self.curl * &self.harm)),
        )
    }
}

/// Three orthogonal Hodge components of an edge flow.
#[derive(Clone, Debug)]
pub struct HodgeComponents {
    /// Gradient (curl-free) part.
    pub grad: EdgeSignal,
    /// Curl (div-free) part.
    pub curl: EdgeSignal,
    /// Harmonic (div- and curl-free) part.
    pub harm: EdgeSignal,
}

impl HodgeComponents {
    /// Fractional energies `(grad, curl, harm)` summing to 1 (or 0 if silent).
    pub fn energy_fractions(&self) -> (f64, f64, f64) {
        let eg = self.grad.energy();
        let ec = self.curl.energy();
        let eh = self.harm.energy();
        let tot = eg + ec + eh;
        if tot <= 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (eg / tot, ec / tot, eh / tot)
        }
    }
}

/// Decompose `f = f_G + f_C + f_H`.
pub fn decompose(sp: &HodgeSpectra, f: &EdgeSignal) -> HodgeComponents {
    let v = f.values();
    HodgeComponents {
        grad: EdgeSignal::new(project(&sp.grad_evecs, v)),
        curl: EdgeSignal::new(project(&sp.curl_evecs, v)),
        harm: EdgeSignal::new(project(&sp.harm_evecs, v)),
    }
}

fn projector(u: &DMatrix<f64>) -> DMatrix<f64> {
    let n = u.nrows();
    if u.ncols() == 0 {
        DMatrix::zeros(n, n)
    } else {
        u * u.transpose()
    }
}

fn project(u: &DMatrix<f64>, f: &DVector<f64>) -> DVector<f64> {
    if u.ncols() == 0 {
        DVector::zeros(f.len())
    } else {
        u * (u.transpose() * f)
    }
}

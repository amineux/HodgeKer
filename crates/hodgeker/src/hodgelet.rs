//! Hodgelet / spectral-wavelet energy features (Alain et al., 2024/2025).
//!
//! We do **not** reimplement the full classification GP of those papers. This
//! module extracts the Hodge-aware wavelet energies used as Euclidean features:
//!
//! ```text
//! W_□(s) = U_□  ψ(s Λ_□)  U_□ᵀ
//! A_□(s) = || W_□(s) f ||₂
//! ```
//!
//! with a Hammond-style band-pass `ψ(t) = t exp(-t²)` plus a low-pass
//! `a(λ) = exp(-γ λ)`. Concatenating `(A_G, A_C, A_H)` across scales yields an
//! isomorphism-friendly signature of an edge flow.

use nalgebra::{DMatrix, DVector};

use crate::ids::EdgeSignal;
use crate::spectra::HodgeSpectra;

/// Wavelet-filter specification.
#[derive(Clone, Debug)]
pub struct HodgeletSpec {
    /// Band-pass scales `s` in `ψ(s λ) = (sλ) exp(-(sλ)²)`.
    pub scales: Vec<f64>,
    /// Low-pass `γ` in `exp(-γ λ)`. `None` skips the scaling function.
    pub lowpass_gamma: Option<f64>,
}

impl Default for HodgeletSpec {
    fn default() -> Self {
        Self {
            scales: vec![0.5, 1.0, 2.0, 4.0],
            lowpass_gamma: Some(1.0),
        }
    }
}

/// Energy features split by Hodge component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HodgeletFeatures {
    /// Gradient-channel energies (low-pass first if present, then scales).
    pub grad: Vec<f64>,
    /// Curl-channel energies.
    pub curl: Vec<f64>,
    /// Harmonic-channel energies.
    pub harm: Vec<f64>,
}

impl HodgeletFeatures {
    /// Flatten in order `grad || curl || harm`.
    pub fn concat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.grad.len() + self.curl.len() + self.harm.len());
        v.extend_from_slice(&self.grad);
        v.extend_from_slice(&self.curl);
        v.extend_from_slice(&self.harm);
        v
    }
}

/// Hodgelet energy signature of an edge signal.
pub fn hodgelet_energy(sp: &HodgeSpectra, f: &EdgeSignal, spec: &HodgeletSpec) -> HodgeletFeatures {
    let v = f.values();
    HodgeletFeatures {
        grad: channel_energy(&sp.grad_evecs, &sp.grad_evals, v, spec),
        curl: channel_energy(&sp.curl_evecs, &sp.curl_evals, v, spec),
        harm: channel_energy(&sp.harm_evecs, &DVector::zeros(sp.n_harm()), v, spec),
    }
}

fn channel_energy(
    evecs: &DMatrix<f64>,
    evals: &DVector<f64>,
    f: &DVector<f64>,
    spec: &HodgeletSpec,
) -> Vec<f64> {
    let mut out = Vec::new();
    if let Some(gamma) = spec.lowpass_gamma {
        out.push(filter_norm(evecs, evals, f, |l| (-gamma * l).exp()));
    }
    for &s in &spec.scales {
        out.push(filter_norm(evecs, evals, f, |l| {
            let t = s * l;
            t * (-t * t).exp()
        }));
    }
    out
}

fn filter_norm(
    evecs: &DMatrix<f64>,
    evals: &DVector<f64>,
    f: &DVector<f64>,
    w: impl Fn(f64) -> f64,
) -> f64 {
    if evecs.ncols() == 0 {
        return 0.0;
    }
    // W f = U diag(w(λ)) Uᵀ f
    let coeff = evecs.transpose() * f;
    let mut wf = DVector::zeros(evecs.nrows());
    for k in 0..evecs.ncols() {
        let amp = w(evals[k]);
        wf += amp * coeff[k] * evecs.column(k);
    }
    wf.norm()
}

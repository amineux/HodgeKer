//! Seeded synthetic edge flows: gradient-, curl-, and mixed-dominated.

use nalgebra::DVector;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::complex::SimplicialComplex2;
use crate::error::{HodgekerError, Result};
use crate::ids::EdgeSignal;
use crate::operators::HodgeOperators;
use crate::spectra::HodgeSpectra;

/// Which synthetic family to draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowKind {
    /// `f = B₁ᵀ φ` for a random (or linear) node potential — curl-free.
    Gradient,
    /// `f = B₂ ψ` for a random (or spatially smooth) face signal — div-free.
    Curl,
    /// Convex combination of gradient, curl, and (if present) harmonic.
    Mixed,
    /// Smooth central vortex on faces plus a weak eastward gradient.
    /// Ocean-current cartoon, not a GCM.
    Ocean,
}

/// Parameters for [`generate`].
#[derive(Clone, Debug)]
pub struct SynthSpec {
    /// Flow family.
    pub kind: FlowKind,
    /// Deterministic seed.
    pub seed: u64,
    /// i.i.d. Gaussian observation noise added **after** the clean flow.
    pub noise_std: f64,
    /// Mix weights `(w_G, w_C, w_H)` used by [`FlowKind::Mixed`].
    pub mix: (f64, f64, f64),
}

impl Default for SynthSpec {
    fn default() -> Self {
        Self {
            kind: FlowKind::Ocean,
            seed: 0,
            noise_std: 0.02,
            mix: (0.2, 0.7, 0.1),
        }
    }
}

/// Draw a seeded edge flow on `sc`.
pub fn generate(
    sc: &SimplicialComplex2,
    ops: &HodgeOperators,
    spec: &SynthSpec,
) -> Result<EdgeSignal> {
    let mut rng = ChaCha8Rng::seed_from_u64(spec.seed);
    let n1 = sc.n_edges();
    if n1 == 0 {
        return Err(HodgekerError::InvalidSimplex("complex has no edges".into()));
    }
    let clean = match spec.kind {
        FlowKind::Gradient => gradient_flow(sc, ops, &mut rng, false),
        FlowKind::Curl => curl_flow(sc, ops, &mut rng, false)?,
        FlowKind::Mixed => {
            let g = gradient_flow(sc, ops, &mut rng, false);
            let c = if sc.n_faces() > 0 {
                curl_flow(sc, ops, &mut rng, true)?
            } else {
                DVector::zeros(n1)
            };
            let h = harmonic_probe(ops, n1);
            spec.mix.0 * g + spec.mix.1 * c + spec.mix.2 * h
        }
        FlowKind::Ocean => ocean_flow(sc, ops)?,
    };
    let mut out = clean;
    if spec.noise_std > 0.0 {
        for i in 0..n1 {
            out[i] += spec.noise_std * std_normal(&mut rng);
        }
    }
    Ok(EdgeSignal::new(out))
}

fn std_normal(rng: &mut ChaCha8Rng) -> f64 {
    // Box–Muller
    let u1: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn gradient_flow(
    sc: &SimplicialComplex2,
    ops: &HodgeOperators,
    rng: &mut ChaCha8Rng,
    linear: bool,
) -> DVector<f64> {
    let n0 = sc.n_vertices();
    let mut phi = DVector::zeros(n0);
    for (i, p) in sc.vertices().iter().enumerate() {
        phi[i] = if linear {
            p.x
        } else {
            std_normal(rng) + 0.3 * p.x + 0.1 * p.y
        };
    }
    ops.grad(&phi)
}

fn curl_flow(
    sc: &SimplicialComplex2,
    ops: &HodgeOperators,
    rng: &mut ChaCha8Rng,
    smooth: bool,
) -> Result<DVector<f64>> {
    let n2 = sc.n_faces();
    if n2 == 0 {
        return Err(HodgekerError::InvalidSimplex(
            "curl flow needs at least one triangle".into(),
        ));
    }
    let mut psi = DVector::zeros(n2);
    for t in 0..n2 {
        let c = sc.face_centroid(crate::ids::FaceId(t));
        psi[t] = if smooth {
            let dx = c.x - 0.5 * (sc.vertices().iter().map(|p| p.x).fold(0.0, f64::max));
            let dy = c.y - 0.5 * (sc.vertices().iter().map(|p| p.y).fold(0.0, f64::max));
            (-(dx * dx + dy * dy) / 8.0).exp()
        } else {
            std_normal(rng)
        };
    }
    Ok(&ops.b2_dense() * psi)
}

fn harmonic_probe(ops: &HodgeOperators, n1: usize) -> DVector<f64> {
    // Cheap stand-in used only by Mixed when we do not want a full eigen:
    // the residual of a random vector after one Jacobi-style damping of L1.
    // The ocean / curl generators do not rely on this.
    let mut v = DVector::from_element(n1, 1.0);
    let l1v = &ops.l1 * &v;
    if l1v.norm() > 1e-12 {
        let step = &ops.l1 * &v;
        v -= step * (1.0 / (ops.l1[(0, 0)].abs() + 1.0));
    }
    let n = v.norm();
    if n > 0.0 {
        v /= n;
    }
    v
}

/// Smooth vortex on faces (`f = B₂ ψ`) plus a weak west→east gradient.
fn ocean_flow(sc: &SimplicialComplex2, ops: &HodgeOperators) -> Result<DVector<f64>> {
    let n2 = sc.n_faces();
    if n2 == 0 {
        return Err(HodgekerError::InvalidSimplex(
            "ocean-like flow needs triangular faces".into(),
        ));
    }
    let (xmin, xmax, ymin, ymax) = bbox(sc);
    let cx = 0.5 * (xmin + xmax);
    let cy = 0.5 * (ymin + ymax);
    let rx = (xmax - xmin).max(1.0);
    let ry = (ymax - ymin).max(1.0);
    let mut psi = DVector::zeros(n2);
    for t in 0..n2 {
        let c = sc.face_centroid(crate::ids::FaceId(t));
        let dx = (c.x - cx) / rx;
        let dy = (c.y - cy) / ry;
        psi[t] = (-(dx * dx + dy * dy) / 0.08).exp();
    }
    let vortex = &ops.b2_dense() * psi;

    let mut phi = DVector::zeros(sc.n_vertices());
    for (i, p) in sc.vertices().iter().enumerate() {
        phi[i] = 0.15 * (p.x - xmin) / rx.max(1e-9);
    }
    let drift = ops.grad(&phi);
    Ok(vortex + drift)
}

fn bbox(sc: &SimplicialComplex2) -> (f64, f64, f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for p in sc.vertices() {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }
    (xmin, xmax, ymin, ymax)
}

/// Convenience: Hodge-split energy fractions of a generated flow.
pub fn energy_report(sp: &HodgeSpectra, f: &EdgeSignal) -> (f64, f64, f64) {
    crate::projectors::decompose(sp, f).energy_fractions()
}

//! Exact GP regression on edges, with a Nyström / inducing-point option.

use nalgebra::{DMatrix, DVector};

use crate::error::{HodgekerError, Result};
use crate::linag::{chol_solve, chol_solve_mat, gather, pseudoinverse_spd, submatrix};

/// Posterior mean / std on a set of target edges.
#[derive(Clone, Debug)]
pub struct GpPrediction {
    /// Posterior mean at the requested indices.
    pub mean: DVector<f64>,
    /// Posterior standard deviation (not variance).
    pub std: DVector<f64>,
    /// Training log marginal likelihood (if computed).
    pub log_marginal: f64,
}

/// Landmark / Nyström approximation of a kernel matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InducingApprox {
    /// Use the exact `N₁ × N₁` kernel.
    Exact,
    /// Nyström from `m` landmark edges: `K ≈ C W⁺ Cᵀ`.
    Nystrom {
        /// Number of landmark edges (evenly strided if `landmarks` is empty).
        m: usize,
    },
}

/// Approximate `K` by Nyström using landmark indices.
pub fn nystrom(k: &DMatrix<f64>, landmarks: &[usize]) -> DMatrix<f64> {
    let n = k.nrows();
    let m = landmarks.len();
    if m == 0 || m >= n {
        return k.clone();
    }
    let mut c = DMatrix::zeros(n, m);
    let mut w = DMatrix::zeros(m, m);
    for (a, &ia) in landmarks.iter().enumerate() {
        for i in 0..n {
            c[(i, a)] = k[(i, ia)];
        }
        for (b, &ib) in landmarks.iter().enumerate() {
            w[(a, b)] = k[(ia, ib)];
        }
    }
    let w_pinv = pseudoinverse_spd(&w, 1e-10);
    &c * w_pinv * c.transpose()
}

/// Evenly strided landmark indices in `0..n`.
pub fn stride_landmarks(n: usize, m: usize) -> Vec<usize> {
    if m == 0 || n == 0 {
        return Vec::new();
    }
    let m = m.min(n);
    (0..m).map(|i| i * n / m).collect()
}

/// GP posterior at `test` given observations `y` on `train`.
///
/// `k` is the prior covariance on **all** edges (or a Nyström proxy).
/// Noise `σ_ε²` is added to the training block.
pub fn predict(
    k: &DMatrix<f64>,
    train: &[usize],
    y: &DVector<f64>,
    test: &[usize],
    noise: f64,
) -> Result<GpPrediction> {
    if y.len() != train.len() {
        return Err(HodgekerError::Dimension(format!(
            "y has {} entries, train has {}",
            y.len(),
            train.len()
        )));
    }
    if train.is_empty() {
        return Err(HodgekerError::Dimension("empty training set".into()));
    }
    let k_tt = submatrix(k, train, train);
    let mut k_noisy = k_tt;
    for i in 0..train.len() {
        k_noisy[(i, i)] += noise.max(0.0);
    }
    let k_st = submatrix(k, test, train);
    let (alpha, logdet) = chol_solve(&k_noisy, y)?;
    let mean = &k_st * &alpha;

    // diag(K_ss - K_st K_tt^{-1} K_ts)
    let v = chol_solve_mat(&k_noisy, &k_st.transpose())?;
    let mut std = DVector::zeros(test.len());
    for i in 0..test.len() {
        let kss = k[(test[i], test[i])];
        let quad: f64 = k_st.row(i).dot(&v.column(i).transpose());
        std[i] = (kss - quad).max(0.0).sqrt();
    }

    let n = train.len() as f64;
    let quad = y.dot(&alpha);
    let log_marginal = -0.5 * quad - 0.5 * logdet - 0.5 * n * (2.0 * std::f64::consts::PI).ln();

    Ok(GpPrediction {
        mean,
        std,
        log_marginal,
    })
}

/// Reconstruct the signal on every edge (train + test).
pub fn predict_all(
    k: &DMatrix<f64>,
    train: &[usize],
    y: &DVector<f64>,
    noise: f64,
) -> Result<GpPrediction> {
    let n = k.nrows();
    let test: Vec<usize> = (0..n).collect();
    predict(k, train, y, &test, noise)
}

/// Mean squared error.
pub fn mse(pred: &DVector<f64>, truth: &DVector<f64>) -> f64 {
    let n = pred.len().max(1) as f64;
    (pred - truth).dot(&(pred - truth)) / n
}

/// Mean absolute error.
pub fn mae(pred: &DVector<f64>, truth: &DVector<f64>) -> f64 {
    let n = pred.len().max(1) as f64;
    pred.iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n
}

/// Subset of `y_full` at `idx`.
pub fn observe(y_full: &DVector<f64>, idx: &[usize]) -> DVector<f64> {
    gather(y_full, idx)
}

/// Seeded Fisher–Yates split of `0..n` into `(train, test)`.
pub fn holdout_split(n: usize, holdout: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    if n < 2 {
        return ((0..n).collect(), Vec::new());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9) ^ 0xA511_E9B3);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        idx.swap(i, j);
    }
    let frac = holdout.clamp(0.05, 0.9);
    let n_test = ((n as f64) * frac).round() as usize;
    let n_test = n_test.clamp(1, n - 1);
    (idx[n_test..].to_vec(), idx[..n_test].to_vec())
}

//! Dense linear-algebra helpers (nalgebra). Sparse incidence lives in `operators`.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::error::{HodgekerError, Result};

/// Convert a `sprs` CSR/CSC matrix into a dense `DMatrix`.
pub fn sprs_to_dense(m: &sprs::CsMat<f64>) -> DMatrix<f64> {
    let mut d = DMatrix::zeros(m.rows(), m.cols());
    for (val, (i, j)) in m.iter() {
        d[(i, j)] = *val;
    }
    d
}

/// Symmetric eigendecomposition with eigenvalues in **ascending** order.
///
/// Eigenvectors are columns of the returned matrix.
pub fn sym_eig(a: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let n = a.nrows();
    if n == 0 {
        return (DVector::zeros(0), DMatrix::zeros(0, 0));
    }
    let se = SymmetricEigen::new(a.clone());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        se.eigenvalues[i]
            .partial_cmp(&se.eigenvalues[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut evals = DVector::zeros(n);
    let mut evecs = DMatrix::zeros(n, n);
    for (k, &i) in order.iter().enumerate() {
        evals[k] = se.eigenvalues[i];
        evecs.set_column(k, &se.eigenvectors.column(i));
    }
    (evals, evecs)
}

/// Relative spectral cutoff from the largest |\lambda|.
pub fn spectral_tol(evals: &DVector<f64>, rel: f64) -> f64 {
    let max_abs = evals.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    (rel * max_abs.max(1.0)).max(1e-12)
}

/// Columns of `evecs` whose eigenvalues satisfy `pred`.
pub fn take_columns(
    evals: &DVector<f64>,
    evecs: &DMatrix<f64>,
    mut pred: impl FnMut(f64) -> bool,
) -> (DVector<f64>, DMatrix<f64>) {
    let idx: Vec<usize> = (0..evals.len()).filter(|&i| pred(evals[i])).collect();
    gather_columns(evals, evecs, &idx)
}

/// Gather eigenpairs by column index.
pub fn gather_columns(
    evals: &DVector<f64>,
    evecs: &DMatrix<f64>,
    idx: &[usize],
) -> (DVector<f64>, DMatrix<f64>) {
    let n = evecs.nrows();
    let m = idx.len();
    let mut lam = DVector::zeros(m);
    let mut u = DMatrix::zeros(n, m);
    for (k, &i) in idx.iter().enumerate() {
        lam[k] = evals[i];
        u.set_column(k, &evecs.column(i));
    }
    (lam, u)
}

/// Frobenius norm.
pub fn frob(a: &DMatrix<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Smallest eigenvalue of a symmetric matrix.
pub fn min_eig_sym(a: &DMatrix<f64>) -> f64 {
    if a.nrows() == 0 {
        return 0.0;
    }
    let (evals, _) = sym_eig(a);
    evals.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Solve `A x = b` with Cholesky, adding a geometric jitter schedule on failure.
///
/// Returns `(x, log det A_jittered)`.
pub fn chol_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<(DVector<f64>, f64)> {
    if a.nrows() != a.ncols() {
        return Err(HodgekerError::LinAlg("chol_solve: non-square".into()));
    }
    if a.nrows() != b.len() {
        return Err(HodgekerError::LinAlg("chol_solve: rhs dimension".into()));
    }
    if a.nrows() == 0 {
        return Ok((DVector::zeros(0), 0.0));
    }
    let n = a.nrows();
    let mut jitter = 0.0;
    for _ in 0..10 {
        let mut m = a.clone();
        if jitter > 0.0 {
            for i in 0..n {
                m[(i, i)] += jitter;
            }
        }
        if let Some(chol) = nalgebra::Cholesky::new(m) {
            let x = chol.solve(b);
            let l = chol.l();
            let mut logdet = 0.0;
            for i in 0..n {
                let d = l[(i, i)];
                if d <= 0.0 {
                    break;
                }
                logdet += 2.0 * d.ln();
            }
            return Ok((x, logdet));
        }
        jitter = if jitter == 0.0 { 1e-12 } else { jitter * 10.0 };
    }
    Err(HodgekerError::LinAlg(
        "Cholesky failed even after jitter; kernel is not SPD".into(),
    ))
}

/// Solve `A X = B` (multiple RHS) via Cholesky with jitter.
pub fn chol_solve_mat(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if a.nrows() == 0 {
        return Ok(DMatrix::zeros(0, b.ncols()));
    }
    let n = a.nrows();
    let mut jitter = 0.0;
    for _ in 0..10 {
        let mut m = a.clone();
        if jitter > 0.0 {
            for i in 0..n {
                m[(i, i)] += jitter;
            }
        }
        if let Some(chol) = nalgebra::Cholesky::new(m) {
            return Ok(chol.solve(b));
        }
        jitter = if jitter == 0.0 { 1e-12 } else { jitter * 10.0 };
    }
    Err(HodgekerError::LinAlg("Cholesky (multi-RHS) failed".into()))
}

/// Symmetric pseudoinverse via eigen-thresholding.
pub fn pseudoinverse_spd(a: &DMatrix<f64>, rel: f64) -> DMatrix<f64> {
    if a.nrows() == 0 {
        return DMatrix::zeros(0, 0);
    }
    let (evals, evecs) = sym_eig(a);
    let tol = spectral_tol(&evals, rel);
    let mut inv = DMatrix::zeros(a.nrows(), a.ncols());
    for k in 0..evals.len() {
        if evals[k].abs() <= tol {
            continue;
        }
        let inv_l = 1.0 / evals[k];
        let u = evecs.column(k);
        inv += inv_l * u * u.transpose();
    }
    inv
}

/// Submatrix `A[rows, cols]`.
pub fn submatrix(a: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(rows.len(), cols.len());
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            out[(i, j)] = a[(r, c)];
        }
    }
    out
}

/// Gather entries `v[idx]`.
pub fn gather(v: &DVector<f64>, idx: &[usize]) -> DVector<f64> {
    DVector::from_iterator(idx.len(), idx.iter().map(|&i| v[i]))
}

/// Scatter `src` into a length-`n` vector at `idx`.
pub fn scatter(n: usize, idx: &[usize], src: &DVector<f64>) -> DVector<f64> {
    let mut out = DVector::zeros(n);
    for (k, &i) in idx.iter().enumerate() {
        out[i] = src[k];
    }
    out
}

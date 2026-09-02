//! Error types.

use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, HodgekerError>;

/// Recoverable HodgeKer failures.
#[derive(Debug, Error)]
pub enum HodgekerError {
    /// A simplex, orientation, or incidence relation is malformed.
    #[error("invalid simplex: {0}")]
    InvalidSimplex(String),
    /// Vector or matrix dimension does not match the complex.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// Filesystem failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parse/serialize failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// Dense linear algebra (Cholesky / eigen) failure.
    #[error("linear algebra: {0}")]
    LinAlg(String),
    /// Text / mesh parse failure.
    #[error("parse: {0}")]
    Parse(String),
    /// A required geometric or combinatorial ingredient is missing.
    #[error("{0}")]
    Other(String),
}

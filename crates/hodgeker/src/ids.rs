//! Strong identifiers and the edge-signal newtype.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::error::{HodgekerError, Result};

/// Vertex index into a [`crate::SimplicialComplex2`].
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct VertexId(pub usize);

impl VertexId {
    /// Zero-based index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Oriented-edge index into a [`crate::SimplicialComplex2`].
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct EdgeId(pub usize);

impl EdgeId {
    /// Zero-based index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Triangle-face index into a [`crate::SimplicialComplex2`].
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FaceId(pub usize);

impl FaceId {
    /// Zero-based index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// A 1-cochain: one real value per oriented edge.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeSignal {
    values: DVector<f64>,
}

impl EdgeSignal {
    /// Wrap a vector; length is checked later against a complex.
    pub fn new(values: DVector<f64>) -> Self {
        Self { values }
    }

    /// Wrap a dense slice.
    pub fn from_slice(values: &[f64]) -> Self {
        Self {
            values: DVector::from_row_slice(values),
        }
    }

    /// Number of edges.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when there are no edges.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow the dense values.
    pub fn values(&self) -> &DVector<f64> {
        &self.values
    }

    /// Mutable dense values.
    pub fn values_mut(&mut self) -> &mut DVector<f64> {
        &mut self.values
    }

    /// Consume and return the dense vector.
    pub fn into_inner(self) -> DVector<f64> {
        self.values
    }

    /// Require `len == n_edges`.
    pub fn expect_len(&self, n_edges: usize) -> Result<()> {
        if self.values.len() == n_edges {
            Ok(())
        } else {
            Err(HodgekerError::Dimension(format!(
                "edge signal has {} entries, expected {}",
                self.values.len(),
                n_edges
            )))
        }
    }

    /// ℓ² energy.
    pub fn energy(&self) -> f64 {
        self.values.dot(&self.values)
    }
}

impl From<DVector<f64>> for EdgeSignal {
    fn from(values: DVector<f64>) -> Self {
        Self::new(values)
    }
}

impl From<Vec<f64>> for EdgeSignal {
    fn from(values: Vec<f64>) -> Self {
        Self::new(DVector::from_vec(values))
    }
}

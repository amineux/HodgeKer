//! HodgeKer: Hodge compositional kernels on simplicial 2-complexes.
//!
//! Original software implementing discrete exterior calculus operators, Hodge
//! decomposition of edge flows, and Matérn / Hodgelet Gaussian-process kernels
//! in the style of Yang, Borovitskiy & Isufi (AISTATS 2024) and the Hodgelet
//! papers of Alain et al. Those papers are the scientific source; this crate
//! does not claim authorship of the mathematics.
//!
//! # Indexing convention
//!
//! Edge flows are 1-cochains: `f([j, i]) = -f([i, j])`. Reference orientations
//! are increasing vertex labels, matching Lim (SIAM Review, 2020) and Yang et al.

#![deny(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod complex;
pub mod demo;
pub mod error;
pub mod gp;
pub mod hodgelet;
pub mod ids;
pub mod io;
pub mod kernel;
pub mod linag;
pub mod operators;
pub mod projectors;
pub mod spectra;
pub mod synth;

pub use complex::{Point, SimplicialComplex2};
pub use demo::{run_ocean_benchmark, BenchmarkConfig, BenchmarkReport};
pub use error::{HodgekerError, Result};
pub use gp::{predict, GpPrediction, InducingApprox};
pub use hodgelet::{hodgelet_energy, HodgeletFeatures, HodgeletSpec};
pub use ids::{EdgeId, EdgeSignal, FaceId, VertexId};
pub use kernel::{
    compositional_matern, edge_matern, graph_matern, line_graph_laplacian, line_graph_spectrum,
    HodgeMaternParams, KernelKind, MaternParams,
};
pub use operators::HodgeOperators;
pub use projectors::{decompose, HodgeComponents, HodgeProjectors};
pub use spectra::{hodge_spectra, HodgeSpectra};
pub use synth::{FlowKind, SynthSpec};

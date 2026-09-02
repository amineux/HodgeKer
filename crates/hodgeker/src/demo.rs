//! Seeded ocean-flow benchmark: Hodge-compositional GP vs graph Matérn.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

use crate::complex::SimplicialComplex2;
use crate::error::Result;
use crate::gp::{mse, observe, predict};
use crate::ids::EdgeSignal;
use crate::kernel::{
    compositional_matern, edge_matern, graph_matern, line_graph_spectrum, HodgeMaternParams,
    MaternParams,
};
use crate::operators::HodgeOperators;
use crate::projectors::{decompose, HodgeProjectors};
use crate::spectra::{hodge_spectra, HodgeSpectra};
use crate::synth::{generate, FlowKind, SynthSpec};

/// Knob set for [`run_ocean_benchmark`].
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Vertex count along x.
    pub nx: usize,
    /// Vertex count along y.
    pub ny: usize,
    /// RNG seed (complex is deterministic; flow and split use the seed).
    pub seed: u64,
    /// Fraction of edges held out for test.
    pub holdout: f64,
    /// Observation noise added to the synthetic flow.
    pub noise_std: f64,
    /// Optional directory for CSV dumps.
    pub out_dir: Option<PathBuf>,
    /// Nyström rank; `None` = exact kernel.
    pub nystrom_m: Option<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            nx: 8,
            ny: 8,
            seed: 42,
            holdout: 0.4,
            noise_std: 0.05,
            out_dir: None,
            nystrom_m: None,
        }
    }
}

/// Held-out reconstruction numbers from one seeded run.
#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkReport {
    /// Vertex grid.
    pub nx: usize,
    /// Vertex grid.
    pub ny: usize,
    /// Seed.
    pub seed: u64,
    /// `N₁`.
    pub n_edges: usize,
    /// `N₂`.
    pub n_faces: usize,
    /// Hodge dims.
    pub n_grad: usize,
    /// Hodge dims.
    pub n_curl: usize,
    /// Hodge dims.
    pub n_harm: usize,
    /// Energy fraction in the gradient component.
    pub energy_grad: f64,
    /// Energy fraction in the curl component.
    pub energy_curl: f64,
    /// Energy fraction in the harmonic component.
    pub energy_harm: f64,
    /// Held-out MSE of the Hodge-compositional GP.
    pub hodge_mse: f64,
    /// Held-out MSE of the line-graph Matérn GP (naive graph kernel).
    pub graph_mse: f64,
    /// Held-out MSE of the non-compositional edge Matérn GP.
    pub edge_mse: f64,
    /// `graph_mse / hodge_mse`.
    pub graph_over_hodge: f64,
    /// Train log-marginal of the selected Hodge kernel.
    pub hodge_log_marginal: f64,
    /// Train log-marginal of the selected line-graph kernel.
    pub graph_log_marginal: f64,
    /// Fitted Hodge hyperparameters.
    pub hodge_params: HodgeMaternParams,
    /// Fitted graph hyperparameters.
    pub graph_params: MaternParams,
    /// Graph kernel noise.
    pub graph_noise: f64,
}

/// Build a triangulated grid, draw an ocean-like (curl-heavy) flow, fit both
/// GPs by log-marginal grid search, and score held-out edges.
pub fn run_ocean_benchmark(cfg: &BenchmarkConfig) -> Result<BenchmarkReport> {
    let sc = SimplicialComplex2::grid(cfg.nx, cfg.ny, true)?;
    let ops = HodgeOperators::from_complex(&sc)?;
    let sp = hodge_spectra(&ops)?;
    let spec = SynthSpec {
        kind: FlowKind::Ocean,
        seed: cfg.seed,
        noise_std: cfg.noise_std,
        mix: (0.0, 1.0, 0.0),
    };
    let flow = generate(&sc, &ops, &spec)?;
    run_on_complex(cfg, &sc, &sp, &flow)
}

fn run_on_complex(
    cfg: &BenchmarkConfig,
    sc: &SimplicialComplex2,
    sp: &HodgeSpectra,
    flow: &EdgeSignal,
) -> Result<BenchmarkReport> {
    let n1 = sc.n_edges();
    let (train, test) = crate::gp::holdout_split(n1, cfg.holdout, cfg.seed);
    let y_full = flow.values();
    let y_train = observe(y_full, &train);
    let y_test = observe(y_full, &test);

    let parts = decompose(sp, flow);
    let (eg, ec, eh) = parts.energy_fractions();

    let (lg_e, lg_u) = line_graph_spectrum(sc);

    let (hodge_params, hodge_k) = fit_hodge(sp, &train, &y_train)?;
    let (graph_params, graph_noise, graph_k) = fit_graph(&lg_e, &lg_u, &train, &y_train)?;
    let (edge_params, edge_noise, edge_k) = fit_shared_edge(sp, &train, &y_train)?;
    let _ = (edge_params, edge_noise);

    let hodge_k = maybe_nystrom(hodge_k, cfg.nystrom_m, n1);
    let graph_k = maybe_nystrom(graph_k, cfg.nystrom_m, n1);
    let edge_k = maybe_nystrom(edge_k, cfg.nystrom_m, n1);

    let hodge_pred = predict(&hodge_k, &train, &y_train, &test, hodge_params.noise)?;
    let graph_pred = predict(&graph_k, &train, &y_train, &test, graph_noise)?;
    let edge_pred = predict(&edge_k, &train, &y_train, &test, edge_noise)?;

    let hodge_mse = mse(&hodge_pred.mean, &y_test);
    let graph_mse = mse(&graph_pred.mean, &y_test);
    let edge_mse = mse(&edge_pred.mean, &y_test);

    if let Some(dir) = &cfg.out_dir {
        fs::create_dir_all(dir)?;
        dump_decomp(dir, sc, flow, &parts)?;
        dump_pred(
            dir,
            sc,
            y_full,
            &train,
            &test,
            &hodge_pred.mean,
            &hodge_pred.std,
            &graph_pred.mean,
            &graph_pred.std,
        )?;
        dump_metrics_table(
            dir,
            hodge_mse,
            graph_mse,
            edge_mse,
            hodge_pred.log_marginal,
            graph_pred.log_marginal,
        )?;
    }

    Ok(BenchmarkReport {
        nx: cfg.nx,
        ny: cfg.ny,
        seed: cfg.seed,
        n_edges: n1,
        n_faces: sc.n_faces(),
        n_grad: sp.n_grad(),
        n_curl: sp.n_curl(),
        n_harm: sp.n_harm(),
        energy_grad: eg,
        energy_curl: ec,
        energy_harm: eh,
        hodge_mse,
        graph_mse,
        edge_mse,
        graph_over_hodge: if hodge_mse > 0.0 {
            graph_mse / hodge_mse
        } else {
            f64::INFINITY
        },
        hodge_log_marginal: hodge_pred.log_marginal,
        graph_log_marginal: graph_pred.log_marginal,
        hodge_params,
        graph_params,
        graph_noise,
    })
}

fn maybe_nystrom(k: DMatrix<f64>, m: Option<usize>, n: usize) -> DMatrix<f64> {
    match m {
        Some(m) if m > 0 && m < n => {
            let marks = crate::gp::stride_landmarks(n, m);
            crate::gp::nystrom(&k, &marks)
        }
        _ => k,
    }
}

fn fit_hodge(
    sp: &HodgeSpectra,
    train: &[usize],
    y: &DVector<f64>,
) -> Result<(HodgeMaternParams, DMatrix<f64>)> {
    let sig_g = [1e-4, 0.05, 0.5];
    let sig_c = [0.25, 1.0, 4.0];
    let sig_h = [1e-4, 0.2];
    let kap_c = [0.5, 1.0, 2.0, 4.0];
    let noise = [1e-3, 1e-2];
    let mut best_lml = f64::NEG_INFINITY;
    let mut best_p = HodgeMaternParams::default();
    let mut best_k = DMatrix::zeros(sp.n_edges(), sp.n_edges());
    for &sg in &sig_g {
        for &sc in &sig_c {
            for &sh in &sig_h {
                for &kc in &kap_c {
                    for &s2 in &noise {
                        let p = HodgeMaternParams {
                            grad: MaternParams::matern32(sg, 1.0),
                            curl: MaternParams::matern32(sc, kc),
                            harm_variance: sh,
                            noise: s2,
                        };
                        let k = compositional_matern(sp, &p);
                        if let Ok(pred) = predict(&k, train, y, train, p.noise) {
                            if pred.log_marginal.is_finite() && pred.log_marginal > best_lml {
                                best_lml = pred.log_marginal;
                                best_p = p;
                                best_k = k;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((best_p, best_k))
}

fn fit_graph(
    evals: &DVector<f64>,
    evecs: &DMatrix<f64>,
    train: &[usize],
    y: &DVector<f64>,
) -> Result<(MaternParams, f64, DMatrix<f64>)> {
    let var = [0.1, 0.5, 1.0, 4.0];
    let kap = [0.5, 1.0, 2.0, 4.0];
    let noise = [1e-3, 1e-2];
    let mut best_lml = f64::NEG_INFINITY;
    let mut best_p = MaternParams::default();
    let mut best_n = 1e-3;
    let mut best_k = DMatrix::zeros(evecs.nrows(), evecs.nrows());
    for &v in &var {
        for &k0 in &kap {
            for &s2 in &noise {
                let p = MaternParams::matern32(v, k0);
                let k = graph_matern(evals, evecs, &p);
                if let Ok(pred) = predict(&k, train, y, train, s2) {
                    if pred.log_marginal.is_finite() && pred.log_marginal > best_lml {
                        best_lml = pred.log_marginal;
                        best_p = p;
                        best_n = s2;
                        best_k = k;
                    }
                }
            }
        }
    }
    Ok((best_p, best_n, best_k))
}

fn fit_shared_edge(
    sp: &HodgeSpectra,
    train: &[usize],
    y: &DVector<f64>,
) -> Result<(MaternParams, f64, DMatrix<f64>)> {
    let var = [0.1, 0.5, 1.0, 4.0];
    let kap = [0.5, 1.0, 2.0, 4.0];
    let noise = [1e-3, 1e-2];
    let mut best_lml = f64::NEG_INFINITY;
    let mut best_p = MaternParams::default();
    let mut best_n = 1e-3;
    let mut best_k = DMatrix::zeros(sp.n_edges(), sp.n_edges());
    for &v in &var {
        for &k0 in &kap {
            for &s2 in &noise {
                let p = MaternParams::matern32(v, k0);
                let k = edge_matern(sp, &p);
                if let Ok(pred) = predict(&k, train, y, train, s2) {
                    if pred.log_marginal.is_finite() && pred.log_marginal > best_lml {
                        best_lml = pred.log_marginal;
                        best_p = p;
                        best_n = s2;
                        best_k = k;
                    }
                }
            }
        }
    }
    Ok((best_p, best_n, best_k))
}

fn dump_decomp(
    dir: &Path,
    sc: &SimplicialComplex2,
    flow: &EdgeSignal,
    parts: &crate::projectors::HodgeComponents,
) -> Result<()> {
    let path = dir.join("decomp.csv");
    let mut f = fs::File::create(path)?;
    writeln!(f, "edge_id,src,dst,x_mid,y_mid,signal,grad,curl,harm")?;
    for e in 0..sc.n_edges() {
        let edge = sc.edges()[e];
        let m = sc.edge_midpoint(crate::ids::EdgeId(e));
        writeln!(
            f,
            "{},{},{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}",
            e,
            edge.src.index(),
            edge.dst.index(),
            m.x,
            m.y,
            flow.values()[e],
            parts.grad.values()[e],
            parts.curl.values()[e],
            parts.harm.values()[e],
        )?;
    }
    Ok(())
}

fn dump_pred(
    dir: &Path,
    sc: &SimplicialComplex2,
    truth: &DVector<f64>,
    train: &[usize],
    test: &[usize],
    hodge_mean: &DVector<f64>,
    hodge_std: &DVector<f64>,
    graph_mean: &DVector<f64>,
    graph_std: &DVector<f64>,
) -> Result<()> {
    let mut is_train = vec![false; sc.n_edges()];
    for &i in train {
        is_train[i] = true;
    }
    // predictions were computed on `test` only — scatter them.
    let mut hm = vec![f64::NAN; sc.n_edges()];
    let mut hs = vec![f64::NAN; sc.n_edges()];
    let mut gm = vec![f64::NAN; sc.n_edges()];
    let mut gs = vec![f64::NAN; sc.n_edges()];
    for (k, &i) in test.iter().enumerate() {
        hm[i] = hodge_mean[k];
        hs[i] = hodge_std[k];
        gm[i] = graph_mean[k];
        gs[i] = graph_std[k];
    }
    let path = dir.join("pred.csv");
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "edge_id,src,dst,x_mid,y_mid,truth,is_train,hodge_mean,hodge_std,graph_mean,graph_std"
    )?;
    for e in 0..sc.n_edges() {
        let edge = sc.edges()[e];
        let m = sc.edge_midpoint(crate::ids::EdgeId(e));
        writeln!(
            f,
            "{},{},{},{:.6},{:.6},{:.8},{},{:.8},{:.8},{:.8},{:.8}",
            e,
            edge.src.index(),
            edge.dst.index(),
            m.x,
            m.y,
            truth[e],
            u8::from(is_train[e]),
            hm[e],
            hs[e],
            gm[e],
            gs[e],
        )?;
    }
    Ok(())
}

fn dump_metrics_table(
    dir: &Path,
    hodge_mse: f64,
    graph_mse: f64,
    edge_mse: f64,
    hodge_lml: f64,
    graph_lml: f64,
) -> Result<()> {
    let path = dir.join("metrics.csv");
    let mut f = fs::File::create(path)?;
    writeln!(f, "model,heldout_mse,train_log_marginal")?;
    writeln!(f, "hodge_matern,{hodge_mse:.10},{hodge_lml:.10}")?;
    writeln!(f, "linegraph_matern,{graph_mse:.10},{graph_lml:.10}")?;
    writeln!(f, "edge_matern,{edge_mse:.10},")?;
    Ok(())
}

impl std::fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HodgeKer ocean-flow demo")?;
        writeln!(
            f,
            "  grid {}×{} verts · {} edges · {} faces · seed {}",
            self.nx, self.ny, self.n_edges, self.n_faces, self.seed
        )?;
        writeln!(
            f,
            "  Hodge dims  grad={}  curl={}  harm={}",
            self.n_grad, self.n_curl, self.n_harm
        )?;
        writeln!(
            f,
            "  energy split  grad={:.3}  curl={:.3}  harm={:.3}",
            self.energy_grad, self.energy_curl, self.energy_harm
        )?;
        writeln!(f, "  held-out MSE")?;
        writeln!(f, "    Hodge-compositional Matérn : {:.6}", self.hodge_mse)?;
        writeln!(f, "    line-graph Matérn          : {:.6}", self.graph_mse)?;
        writeln!(f, "    non-HC edge Matérn         : {:.6}", self.edge_mse)?;
        writeln!(
            f,
            "    line-graph / Hodge ratio   : {:.3}",
            self.graph_over_hodge
        )?;
        Ok(())
    }
}

/// Projectors for a complex — used by the CLI `decompose` command.
pub fn projectors_for(
    sc: &SimplicialComplex2,
) -> Result<(HodgeOperators, HodgeSpectra, HodgeProjectors)> {
    let ops = HodgeOperators::from_complex(sc)?;
    let sp = hodge_spectra(&ops)?;
    let p = HodgeProjectors::from_spectra(&sp);
    Ok((ops, sp, p))
}

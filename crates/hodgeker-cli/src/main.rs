//! `hodgeker` command-line interface.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use hodgeker::complex::SimplicialComplex2;
use hodgeker::demo::{projectors_for, run_ocean_benchmark, BenchmarkConfig};
use hodgeker::gp::{observe, predict, InducingApprox};
use hodgeker::hodgelet::{hodgelet_energy, HodgeletSpec};
use hodgeker::io::{load_complex, load_signal, save_json, save_signal};
use hodgeker::kernel::{
    assemble, line_graph_spectrum, HodgeMaternParams, KernelKind, MaternParams,
};
use hodgeker::projectors::decompose;
use hodgeker::spectra::hodge_spectra;
use hodgeker::synth::{generate, FlowKind, SynthSpec};

#[derive(Parser, Debug)]
#[command(
    name = "hodgeker",
    version,
    about = "Hodge compositional kernels on simplicial 2-complexes",
    long_about = "Build an SC₂, split an edge flow into grad ⊕ curl ⊕ harmonic, \
                  and fit Matérn GPs that respect the Hodge decomposition.\n\n\
                  Graphs lie about flows: the 1-skeleton Laplacian cannot see \
                  circulation. This CLI exists so you can measure that, not \
                  just quote it."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Emit a triangulated grid (JSON) and optional synthetic flow.
    Build {
        /// Output complex JSON.
        #[arg(long, default_value = "complex.json")]
        out: PathBuf,
        /// Vertices along x.
        #[arg(long, default_value_t = 8)]
        nx: usize,
        /// Vertices along y.
        #[arg(long, default_value_t = 8)]
        ny: usize,
        /// Skip triangular faces (graph 1-skeleton only).
        #[arg(long)]
        no_faces: bool,
        /// Optional synthetic signal path.
        #[arg(long)]
        signal: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = FlowArg::Ocean)]
        kind: FlowArg,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Hodge-split an edge signal.
    Decompose {
        /// Complex JSON / OFF / OBJ.
        #[arg(long)]
        complex: PathBuf,
        /// Signal CSV (one value per edge, or `edge,value`).
        #[arg(long)]
        signal: PathBuf,
        /// Output directory for decomp.csv.
        #[arg(long, default_value = "decomp_out")]
        out: PathBuf,
    },
    /// Fit a GP on a subset of edges and predict with uncertainty.
    Fit {
        #[arg(long)]
        complex: PathBuf,
        #[arg(long)]
        signal: PathBuf,
        #[arg(long, value_enum, default_value_t = KernelArg::Hodge)]
        kernel: KernelArg,
        #[arg(long, default_value_t = 0.4)]
        holdout: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "fit_out")]
        out: PathBuf,
        /// Nyström landmarks (omit for exact).
        #[arg(long)]
        nystrom: Option<usize>,
    },
    /// Ocean-current cartoon: Hodge GP vs graph Matérn on a curl-heavy flow.
    Demo {
        #[arg(long, default_value_t = 8)]
        nx: usize,
        #[arg(long, default_value_t = 8)]
        ny: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 0.4)]
        holdout: f64,
        #[arg(long, default_value_t = 0.05)]
        noise: f64,
        #[arg(long, default_value = "demo_out")]
        out: PathBuf,
        #[arg(long)]
        nystrom: Option<usize>,
    },
    /// Hodgelet energy features of a signal.
    Hodgelets {
        #[arg(long)]
        complex: PathBuf,
        #[arg(long)]
        signal: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FlowArg {
    Gradient,
    Curl,
    Mixed,
    Ocean,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum KernelArg {
    Hodge,
    Edge,
    Graph,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build {
            out,
            nx,
            ny,
            no_faces,
            signal,
            kind,
            seed,
        } => cmd_build(out, nx, ny, no_faces, signal, kind, seed),
        Cmd::Decompose {
            complex,
            signal,
            out,
        } => cmd_decompose(complex, signal, out),
        Cmd::Fit {
            complex,
            signal,
            kernel,
            holdout,
            seed,
            out,
            nystrom,
        } => cmd_fit(complex, signal, kernel, holdout, seed, out, nystrom),
        Cmd::Demo {
            nx,
            ny,
            seed,
            holdout,
            noise,
            out,
            nystrom,
        } => cmd_demo(nx, ny, seed, holdout, noise, out, nystrom),
        Cmd::Hodgelets { complex, signal } => cmd_hodgelets(complex, signal),
    }
}

fn cmd_build(
    out: PathBuf,
    nx: usize,
    ny: usize,
    no_faces: bool,
    signal: Option<PathBuf>,
    kind: FlowArg,
    seed: u64,
) -> Result<()> {
    let sc = SimplicialComplex2::grid(nx, ny, !no_faces)?;
    save_json(&out, &sc)?;
    eprintln!(
        "wrote {}  (N0={} N1={} N2={})",
        out.display(),
        sc.n_vertices(),
        sc.n_edges(),
        sc.n_faces()
    );
    if let Some(sig_path) = signal {
        let ops = hodgeker::HodgeOperators::from_complex(&sc)?;
        let spec = SynthSpec {
            kind: match kind {
                FlowArg::Gradient => FlowKind::Gradient,
                FlowArg::Curl => FlowKind::Curl,
                FlowArg::Mixed => FlowKind::Mixed,
                FlowArg::Ocean => FlowKind::Ocean,
            },
            seed,
            noise_std: 0.0,
            mix: (0.15, 0.75, 0.1),
        };
        let f = generate(&sc, &ops, &spec)?;
        save_signal(&sig_path, &f)?;
        eprintln!("wrote {}", sig_path.display());
    }
    Ok(())
}

fn cmd_decompose(complex: PathBuf, signal: PathBuf, out: PathBuf) -> Result<()> {
    let mut sc = load_complex(&complex)?;
    sc.reindex();
    let flow = load_signal(&signal)?;
    flow.expect_len(sc.n_edges())?;
    let (_, sp, _) = projectors_for(&sc)?;
    let parts = decompose(&sp, &flow);
    let (eg, ec, eh) = parts.energy_fractions();
    std::fs::create_dir_all(&out)?;
    let path = out.join("decomp.csv");
    let mut w = String::from("edge_id,src,dst,x_mid,y_mid,signal,grad,curl,harm\n");
    for e in 0..sc.n_edges() {
        let edge = sc.edges()[e];
        let m = sc.edge_midpoint(hodgeker::EdgeId(e));
        w.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8}\n",
            e,
            edge.src.index(),
            edge.dst.index(),
            m.x,
            m.y,
            flow.values()[e],
            parts.grad.values()[e],
            parts.curl.values()[e],
            parts.harm.values()[e],
        ));
    }
    std::fs::write(&path, w)?;
    println!("Hodge split  grad={eg:.4}  curl={ec:.4}  harm={eh:.4}");
    println!(
        "dims         grad={}  curl={}  harm={}",
        sp.n_grad(),
        sp.n_curl(),
        sp.n_harm()
    );
    println!("wrote {}", path.display());
    Ok(())
}

fn cmd_fit(
    complex: PathBuf,
    signal: PathBuf,
    kernel: KernelArg,
    holdout: f64,
    seed: u64,
    out: PathBuf,
    nystrom: Option<usize>,
) -> Result<()> {
    let mut sc = load_complex(&complex)?;
    sc.reindex();
    let flow = load_signal(&signal)?;
    flow.expect_len(sc.n_edges())?;
    let ops = hodgeker::HodgeOperators::from_complex(&sc)?;
    let sp = hodge_spectra(&ops)?;
    let n1 = sc.n_edges();
    let (train, test) = hodgeker::gp::holdout_split(n1, holdout, seed);
    let y_train = observe(flow.values(), &train);
    let y_test = observe(flow.values(), &test);

    let hodge = HodgeMaternParams {
        grad: MaternParams::matern32(0.05, 1.0),
        curl: MaternParams::matern32(1.0, 1.0),
        harm_variance: 0.05,
        noise: 1e-3,
    };
    let shared = MaternParams::matern32(1.0, 1.0);
    let (lg_e, lg_u) = line_graph_spectrum(&sc);
    let kind = match kernel {
        KernelArg::Hodge => KernelKind::HodgeMatern,
        KernelArg::Edge => KernelKind::EdgeMatern,
        KernelArg::Graph => KernelKind::GraphMatern,
    };
    let mut k = assemble(kind, &sp, &hodge, &shared, &lg_e, &lg_u)?;
    if let Some(m) = nystrom {
        let _ = InducingApprox::Nystrom { m };
        let marks = hodgeker::gp::stride_landmarks(n1, m);
        k = hodgeker::gp::nystrom(&k, &marks);
    }
    let noise = match kernel {
        KernelArg::Hodge => hodge.noise,
        _ => 1e-3,
    };
    let pred = predict(&k, &train, &y_train, &test, noise)?;
    let err = hodgeker::gp::mse(&pred.mean, &y_test);
    std::fs::create_dir_all(&out)?;
    let mut csv = String::from("edge_id,truth,mean,std,split\n");
    for (k_i, &e) in test.iter().enumerate() {
        csv.push_str(&format!(
            "{},{:.8},{:.8},{:.8},test\n",
            e,
            flow.values()[e],
            pred.mean[k_i],
            pred.std[k_i]
        ));
    }
    for &e in &train {
        csv.push_str(&format!(
            "{},{:.8},{:.8},0.0,train\n",
            e,
            flow.values()[e],
            flow.values()[e]
        ));
    }
    let pred_path = out.join("pred.csv");
    std::fs::write(&pred_path, csv)?;
    println!(
        "kernel {:?}\nheld-out MSE = {:.6}\ntrain log-marginal = {:.4}",
        kernel, err, pred.log_marginal
    );
    println!("wrote {}", pred_path.display());
    Ok(())
}

fn cmd_demo(
    nx: usize,
    ny: usize,
    seed: u64,
    holdout: f64,
    noise: f64,
    out: PathBuf,
    nystrom: Option<usize>,
) -> Result<()> {
    if nx < 3 || ny < 3 {
        bail!("demo grid needs nx, ny ≥ 3");
    }
    let cfg = BenchmarkConfig {
        nx,
        ny,
        seed,
        holdout,
        noise_std: noise,
        out_dir: Some(out.clone()),
        nystrom_m: nystrom,
    };
    let report = run_ocean_benchmark(&cfg).context("ocean benchmark")?;
    print!("{report}");
    let metrics = out.join("metrics.json");
    std::fs::write(&metrics, serde_json::to_string_pretty(&report)?)?;
    println!("CSV/JSON written under {}", out.display());
    if report.hodge_mse < report.graph_mse {
        println!("Hodge GP beat the line-graph Matérn baseline on this curl-heavy flow.");
    } else {
        eprintln!("warning: Hodge MSE did not beat line-graph MSE on this seed — try another seed or a finer grid.");
    }
    Ok(())
}

fn cmd_hodgelets(complex: PathBuf, signal: PathBuf) -> Result<()> {
    let mut sc = load_complex(&complex)?;
    sc.reindex();
    let flow = load_signal(&signal)?;
    flow.expect_len(sc.n_edges())?;
    let ops = hodgeker::HodgeOperators::from_complex(&sc)?;
    let sp = hodge_spectra(&ops)?;
    let feat = hodgelet_energy(&sp, &flow, &HodgeletSpec::default());
    println!("grad {:?}", feat.grad);
    println!("curl {:?}", feat.curl);
    println!("harm {:?}", feat.harm);
    println!("concat {:?}", feat.concat());
    Ok(())
}

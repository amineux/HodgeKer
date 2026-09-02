//! The hiring-manager test: Hodge GP beats a graph Matérn on curl-heavy flow.

use hodgeker::demo::{run_ocean_benchmark, BenchmarkConfig};

#[test]
fn curl_heavy_hodge_beats_graph_matern() {
    let report = run_ocean_benchmark(&BenchmarkConfig {
        nx: 6,
        ny: 6,
        seed: 7,
        holdout: 0.4,
        noise_std: 0.04,
        out_dir: None,
        nystrom_m: None,
    })
    .expect("benchmark");

    eprintln!("{report}");
    assert!(
        report.energy_curl > 0.75,
        "ocean cartoon should be curl-dominated, curl energy = {}",
        report.energy_curl
    );
    assert!(
        report.hodge_mse < report.graph_mse,
        "Hodge MSE {:.6} should beat line-graph MSE {:.6}",
        report.hodge_mse,
        report.graph_mse
    );
    assert!(
        report.graph_over_hodge > 1.05,
        "expected a clear gap, ratio = {}",
        report.graph_over_hodge
    );
}

#[test]
fn nystrom_kernel_is_usable() {
    let report = run_ocean_benchmark(&BenchmarkConfig {
        nx: 6,
        ny: 6,
        seed: 7,
        holdout: 0.4,
        noise_std: 0.04,
        out_dir: None,
        nystrom_m: Some(48),
    })
    .expect("nystrom benchmark");
    eprintln!("{report}");
    assert!(report.hodge_mse.is_finite() && report.graph_mse.is_finite());
    assert!(
        report.hodge_mse < report.graph_mse,
        "Nyström Hodge {:.4} should still beat Nyström line-graph {:.4}",
        report.hodge_mse,
        report.graph_mse
    );
}

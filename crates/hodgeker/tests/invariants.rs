//! Incidence identities, Hodge orthogonality, projector algebra, kernel PSD.

use approx::assert_abs_diff_eq;
use hodgeker::complex::SimplicialComplex2;
use hodgeker::kernel::{compositional_matern, HodgeMaternParams, MaternParams};
use hodgeker::linag::{frob, min_eig_sym};
use hodgeker::operators::HodgeOperators;
use hodgeker::projectors::{decompose, HodgeProjectors};
use hodgeker::spectra::hodge_spectra;
use hodgeker::synth::{generate, FlowKind, SynthSpec};

fn operators_on(sc: &SimplicialComplex2) -> (HodgeOperators, hodgeker::HodgeSpectra) {
    let ops = HodgeOperators::from_complex(sc).unwrap();
    let sp = hodge_spectra(&ops).unwrap();
    (ops, sp)
}

#[test]
fn chain_identity_triangle() {
    let sc = SimplicialComplex2::triangle();
    let ops = HodgeOperators::from_complex(&sc).unwrap();
    assert!(ops.chain_identity_residual() < 1e-12);
}

#[test]
fn chain_identity_grid() {
    let sc = SimplicialComplex2::grid(4, 5, true).unwrap();
    let ops = HodgeOperators::from_complex(&sc).unwrap();
    assert!(
        ops.chain_identity_residual() < 1e-10,
        "||B1 B2||_F = {}",
        ops.chain_identity_residual()
    );
}

#[test]
fn chain_identity_graph_no_faces() {
    let sc = SimplicialComplex2::grid(4, 4, false).unwrap();
    let ops = HodgeOperators::from_complex(&sc).unwrap();
    assert_eq!(sc.n_faces(), 0);
    assert!(ops.chain_identity_residual() < 1e-15);
}

#[test]
fn hodge_dims_disk() {
    // A triangulated grid is a topological disk: β1 = 0, n_grad = n0 - 1,
    // n_curl = n2, and n_grad + n_curl + n_harm = n1.
    let sc = SimplicialComplex2::grid(5, 4, true).unwrap();
    let (ops, sp) = operators_on(&sc);
    let _ = ops;
    let n0 = sc.n_vertices();
    let n1 = sc.n_edges();
    let n2 = sc.n_faces();
    assert_eq!(sp.n_grad() + sp.n_curl() + sp.n_harm(), n1);
    assert_eq!(sp.n_grad(), n0 - 1, "connected ⇒ dim im B1^T = n0-1");
    assert_eq!(sp.n_curl(), n2, "disk ⇒ B2 injective, dim im B2 = n2");
    assert_eq!(sp.n_harm(), 0, "disk ⇒ β1 = 0");
}

#[test]
fn hodge_hole_has_harmonic() {
    let sc = SimplicialComplex2::grid_with_hole(6, 6, (2, 4, 2, 4)).unwrap();
    let (_ops, sp) = operators_on(&sc);
    assert!(
        sp.n_harm() >= 1,
        "a grid with a face-hole should have β1 ≥ 1, got {}",
        sp.n_harm()
    );
    assert_eq!(sp.n_grad() + sp.n_curl() + sp.n_harm(), sc.n_edges());
}

#[test]
fn projectors_idempotent_and_orthogonal() {
    let sc = SimplicialComplex2::grid(5, 5, true).unwrap();
    let (_ops, sp) = operators_on(&sc);
    let p = HodgeProjectors::from_spectra(&sp);
    let (ig, ic, ih) = p.idempotence_residuals();
    let (ogc, ogh, och) = p.orthogonality_residuals();
    let n = sc.n_edges() as f64;
    let scale = n.max(1.0);
    assert!(ig / scale < 1e-8, "P_G^2 - P_G = {ig}");
    assert!(ic / scale < 1e-8, "P_C^2 - P_C = {ic}");
    assert!(ih / scale < 1e-8, "P_H^2 - P_H = {ih}");
    assert!(ogc / scale < 1e-8, "P_G P_C = {ogc}");
    assert!(ogh / scale < 1e-8, "P_G P_H = {ogh}");
    assert!(och / scale < 1e-8, "P_C P_H = {och}");

    let sum = &p.grad + &p.curl + &p.harm;
    let ident = nalgebra::DMatrix::<f64>::identity(sc.n_edges(), sc.n_edges());
    assert!(
        frob(&(&sum - &ident)) / scale < 1e-7,
        "P_G+P_C+P_H should be I"
    );
}

#[test]
fn hodge_components_orthogonal() {
    let sc = SimplicialComplex2::grid(6, 5, true).unwrap();
    let (ops, sp) = operators_on(&sc);
    let f = generate(
        &sc,
        &ops,
        &SynthSpec {
            kind: FlowKind::Mixed,
            seed: 11,
            noise_std: 0.0,
            mix: (0.4, 0.4, 0.2),
        },
    )
    .unwrap();
    let parts = decompose(&sp, &f);
    let g = parts.grad.values();
    let c = parts.curl.values();
    let h = parts.harm.values();
    assert_abs_diff_eq!(g.dot(c), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(g.dot(h), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(c.dot(h), 0.0, epsilon = 1e-8);
    let recon = g + c + h;
    assert!((recon - f.values()).norm() < 1e-8 * (1.0 + f.values().norm()));

    // Gradient is curl-free; curl is div-free.
    let curl_g = ops.curl(g);
    let div_c = ops.div(c);
    assert!(curl_g.norm() < 1e-8 * (1.0 + g.norm()));
    assert!(div_c.norm() < 1e-8 * (1.0 + c.norm()));
}

#[test]
fn compositional_kernel_is_psd() {
    let sc = SimplicialComplex2::grid(4, 4, true).unwrap();
    let (_ops, sp) = operators_on(&sc);
    let p = HodgeMaternParams {
        grad: MaternParams::matern32(0.7, 1.2),
        curl: MaternParams::matern32(1.4, 0.8),
        harm_variance: 0.3,
        noise: 0.0,
    };
    let k = compositional_matern(&sp, &p);
    let min_l = min_eig_sym(&k);
    assert!(min_l >= -1e-8, "compositional Matérn min eig = {min_l}");
    assert_eq!(k.nrows(), sc.n_edges());
    // Symmetric.
    assert!(frob(&(k.clone() - k.transpose())) < 1e-12);
}

#[test]
fn json_roundtrip() {
    let sc = SimplicialComplex2::triangle();
    let j = hodgeker::io::ComplexJson::from_complex(&sc);
    let sc2 = j.to_complex().unwrap();
    assert_eq!(sc.n_vertices(), sc2.n_vertices());
    assert_eq!(sc.n_edges(), sc2.n_edges());
    assert_eq!(sc.n_faces(), sc2.n_faces());
}

#[test]
fn off_triangle() {
    let off = "\
OFF
3 1 3
0 0 0
1 0 0
0 1 0
3 0 1 2
";
    let sc = hodgeker::io::load_off(off).unwrap();
    assert_eq!(sc.n_vertices(), 3);
    assert_eq!(sc.n_faces(), 1);
    assert_eq!(sc.n_edges(), 3);
}

#[test]
fn hodgelets_nonzero_on_curl() {
    let sc = SimplicialComplex2::grid(5, 5, true).unwrap();
    let (ops, sp) = operators_on(&sc);
    let f = generate(
        &sc,
        &ops,
        &SynthSpec {
            kind: FlowKind::Curl,
            seed: 3,
            noise_std: 0.0,
            mix: (0.0, 1.0, 0.0),
        },
    )
    .unwrap();
    let feat = hodgeker::hodgelet_energy(&sp, &f, &hodgeker::HodgeletSpec::default());
    let curl_e: f64 = feat.curl.iter().sum();
    let grad_e: f64 = feat.grad.iter().sum();
    assert!(curl_e > grad_e);
}

#[test]
fn line_graph_laplacian_psd() {
    let sc = SimplicialComplex2::grid(4, 4, true).unwrap();
    let l = hodgeker::kernel::line_graph_laplacian(&sc);
    assert_eq!(l.nrows(), sc.n_edges());
    let min_l = hodgeker::linag::min_eig_sym(&l);
    assert!(min_l >= -1e-8, "line-graph L min eig = {min_l}");
    assert!(frob(&(l.clone() - l.transpose())) < 1e-12);
}

#[test]
fn nystrom_reproduces_exact_when_m_eq_n() {
    let km = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 1.0]);
    let approx = hodgeker::gp::nystrom(&km, &[0, 1]);
    assert!(frob(&(approx - km)) < 1e-8);
}

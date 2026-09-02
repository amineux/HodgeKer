//! Mesh / complex IO: JSON (`SC2`), OFF, OBJ, and signal CSV.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::complex::{Point, SimplicialComplex2};
use crate::error::{HodgekerError, Result};
use crate::ids::EdgeSignal;

/// On-disk JSON schema for an `SC₂`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplexJson {
    /// Vertices as length-2 or -3 arrays.
    pub vertices: Vec<Vec<f64>>,
    /// Optional edges `[i, j]`. Face boundaries are added automatically.
    #[serde(default)]
    pub edges: Vec<Vec<usize>>,
    /// Triangles `[i, j, k]`.
    #[serde(default)]
    pub faces: Vec<Vec<usize>>,
}

impl ComplexJson {
    /// Convert to a complex.
    pub fn to_complex(&self) -> Result<SimplicialComplex2> {
        let mut vertices = Vec::with_capacity(self.vertices.len());
        for (i, v) in self.vertices.iter().enumerate() {
            match v.as_slice() {
                [x, y] => vertices.push(Point::xy(*x, *y)),
                [x, y, z] => vertices.push(Point::xyz(*x, *y, *z)),
                _ => {
                    return Err(HodgekerError::Parse(format!(
                        "vertex {i} must have 2 or 3 coordinates"
                    )));
                }
            }
        }
        let mut edges = Vec::new();
        for (i, e) in self.edges.iter().enumerate() {
            if e.len() != 2 {
                return Err(HodgekerError::Parse(format!(
                    "edge {i} must have 2 endpoints"
                )));
            }
            edges.push((e[0], e[1]));
        }
        let mut faces = Vec::new();
        for (i, f) in self.faces.iter().enumerate() {
            if f.len() != 3 {
                return Err(HodgekerError::Parse(format!(
                    "face {i} must be a triangle (got {} verts)",
                    f.len()
                )));
            }
            faces.push([f[0], f[1], f[2]]);
        }
        SimplicialComplex2::new(vertices, edges, faces)
    }

    /// Serialize a complex.
    pub fn from_complex(sc: &SimplicialComplex2) -> Self {
        Self {
            vertices: sc.vertices().iter().map(|p| vec![p.x, p.y, p.z]).collect(),
            edges: sc
                .edges()
                .iter()
                .map(|e| vec![e.src.index(), e.dst.index()])
                .collect(),
            faces: sc
                .faces()
                .iter()
                .map(|f| vec![f.verts[0].index(), f.verts[1].index(), f.verts[2].index()])
                .collect(),
        }
    }
}

/// Load JSON / OFF / OBJ by extension.
pub fn load_complex(path: &Path) -> Result<SimplicialComplex2> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "json" => load_json(path),
        "off" => load_off(&fs::read_to_string(path)?),
        "obj" => load_obj(&fs::read_to_string(path)?),
        _ => Err(HodgekerError::Parse(format!(
            "unknown complex extension '{ext}' (use json, off, obj)"
        ))),
    }
}

/// Write a complex as JSON.
pub fn save_json(path: &Path, sc: &SimplicialComplex2) -> Result<()> {
    let j = ComplexJson::from_complex(sc);
    fs::write(path, serde_json::to_string_pretty(&j)?)?;
    Ok(())
}

/// Read JSON.
pub fn load_json(path: &Path) -> Result<SimplicialComplex2> {
    let s = fs::read_to_string(path)?;
    let j: ComplexJson = serde_json::from_str(&s)?;
    j.to_complex()
}

/// Minimal OFF parser (triangles; quads split on the diagonal).
pub fn load_off(text: &str) -> Result<SimplicialComplex2> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let magic = lines
        .next()
        .ok_or_else(|| HodgekerError::Parse("empty OFF".into()))?;
    if !magic.eq_ignore_ascii_case("off") && !magic.to_ascii_uppercase().starts_with("OFF") {
        return Err(HodgekerError::Parse("missing OFF header".into()));
    }
    let header = if magic.eq_ignore_ascii_case("off") {
        lines
            .next()
            .ok_or_else(|| HodgekerError::Parse("OFF missing counts".into()))?
    } else {
        magic.trim_start_matches(|c: char| c.is_ascii_alphabetic())
    };
    let counts: Vec<&str> = header.split_whitespace().collect();
    if counts.len() < 2 {
        return Err(HodgekerError::Parse("OFF counts line malformed".into()));
    }
    let nv: usize = counts[0]
        .parse()
        .map_err(|_| HodgekerError::Parse("OFF n_vertices".into()))?;
    let nf: usize = counts[1]
        .parse()
        .map_err(|_| HodgekerError::Parse("OFF n_faces".into()))?;
    let mut vertices = Vec::with_capacity(nv);
    for _ in 0..nv {
        let line = lines
            .next()
            .ok_or_else(|| HodgekerError::Parse("OFF truncated vertices".into()))?;
        let nums: Vec<f64> = line
            .split_whitespace()
            .take(3)
            .map(|s| s.parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| HodgekerError::Parse("OFF vertex float".into()))?;
        match nums.as_slice() {
            [x, y, z] => vertices.push(Point::xyz(*x, *y, *z)),
            [x, y] => vertices.push(Point::xy(*x, *y)),
            _ => return Err(HodgekerError::Parse("OFF vertex needs 2–3 coords".into())),
        }
    }
    let mut faces = Vec::new();
    for _ in 0..nf {
        let line = lines
            .next()
            .ok_or_else(|| HodgekerError::Parse("OFF truncated faces".into()))?;
        let nums: Vec<usize> = line
            .split_whitespace()
            .map(|s| s.parse::<usize>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| HodgekerError::Parse("OFF face int".into()))?;
        if nums.is_empty() {
            continue;
        }
        let n = nums[0];
        let vs = &nums[1..];
        if vs.len() < n {
            return Err(HodgekerError::Parse("OFF face vertex count".into()));
        }
        match n {
            3 => faces.push([vs[0], vs[1], vs[2]]),
            4 => {
                faces.push([vs[0], vs[1], vs[2]]);
                faces.push([vs[0], vs[2], vs[3]]);
            }
            _ => {
                return Err(HodgekerError::Parse(format!(
                    "OFF face with {n} verts not supported (3 or 4)"
                )));
            }
        }
    }
    SimplicialComplex2::new(vertices, Vec::new(), faces)
}

/// Minimal OBJ parser (`v` and triangular `f`).
pub fn load_obj(text: &str) -> Result<SimplicialComplex2> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let x: f64 = it
                    .next()
                    .ok_or_else(|| HodgekerError::Parse("OBJ v x".into()))?
                    .parse()
                    .map_err(|_| HodgekerError::Parse("OBJ v x".into()))?;
                let y: f64 = it
                    .next()
                    .ok_or_else(|| HodgekerError::Parse("OBJ v y".into()))?
                    .parse()
                    .map_err(|_| HodgekerError::Parse("OBJ v y".into()))?;
                let z: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                vertices.push(Point::xyz(x, y, z));
            }
            Some("f") => {
                let mut ids = Vec::new();
                for tok in it {
                    let idx = tok
                        .split('/')
                        .next()
                        .unwrap_or(tok)
                        .parse::<i32>()
                        .map_err(|_| HodgekerError::Parse("OBJ f index".into()))?;
                    let id = if idx < 0 {
                        vertices.len() as i32 + idx
                    } else {
                        idx - 1
                    };
                    if id < 0 {
                        return Err(HodgekerError::Parse("OBJ f index underflow".into()));
                    }
                    ids.push(id as usize);
                }
                match ids.as_slice() {
                    [a, b, c] => faces.push([*a, *b, *c]),
                    [a, b, c, d] => {
                        faces.push([*a, *b, *c]);
                        faces.push([*a, *c, *d]);
                    }
                    _ => {
                        return Err(HodgekerError::Parse(
                            "OBJ faces must be triangles or quads".into(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    SimplicialComplex2::new(vertices, Vec::new(), faces)
}

/// Load a signal: one float per line, or `edge,value` CSV.
pub fn load_signal(path: &Path) -> Result<EdgeSignal> {
    let text = fs::read_to_string(path)?;
    parse_signal(&text)
}

/// Parse a signal from text.
pub fn parse_signal(text: &str) -> Result<EdgeSignal> {
    let mut pairs: Vec<(usize, f64)> = Vec::new();
    let mut sequential = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("edge") || lower.starts_with("id") || lower.starts_with("value") {
            continue;
        }
        let parts: Vec<&str> = line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .collect();
        match parts.as_slice() {
            [v] => {
                let x: f64 = v.parse().map_err(|_| {
                    HodgekerError::Parse(format!("line {}: not a float", lineno + 1))
                })?;
                sequential.push(x);
            }
            [i, v] => {
                let idx: usize = i.parse().map_err(|_| {
                    HodgekerError::Parse(format!("line {}: bad edge id", lineno + 1))
                })?;
                let x: f64 = v.parse().map_err(|_| {
                    HodgekerError::Parse(format!("line {}: not a float", lineno + 1))
                })?;
                pairs.push((idx, x));
            }
            _ => {
                return Err(HodgekerError::Parse(format!(
                    "line {}: expected `value` or `edge,value`",
                    lineno + 1
                )));
            }
        }
    }
    if !pairs.is_empty() {
        pairs.sort_by_key(|p| p.0);
        let n = pairs.last().map(|p| p.0 + 1).unwrap_or(0);
        let mut vals = vec![0.0; n];
        for (i, x) in pairs {
            if i >= vals.len() {
                vals.resize(i + 1, 0.0);
            }
            vals[i] = x;
        }
        return Ok(EdgeSignal::from(vals));
    }
    Ok(EdgeSignal::from(sequential))
}

/// Write one float per line.
pub fn save_signal(path: &Path, f: &EdgeSignal) -> Result<()> {
    let mut s = String::new();
    for x in f.values().iter() {
        s.push_str(&format!("{x:.10}\n"));
    }
    fs::write(path, s)?;
    Ok(())
}

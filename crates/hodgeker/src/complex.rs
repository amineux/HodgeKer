//! Simplicial 2-complexes (vertices, oriented edges, triangular faces).
//!
//! Graphs are the 1-skeleton: construct with edges and no faces.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{HodgekerError, Result};
use crate::ids::{EdgeId, FaceId, VertexId};

/// Embedded vertex in R³ (z = 0 for planar complexes).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// x-coordinate.
    pub x: f64,
    /// y-coordinate.
    pub y: f64,
    /// z-coordinate.
    pub z: f64,
}

impl Point {
    /// 2-D point in the z = 0 plane.
    pub fn xy(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    /// 3-D point.
    pub fn xyz(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean midpoint with `other`.
    pub fn midpoint(self, other: Self) -> Self {
        Self {
            x: 0.5 * (self.x + other.x),
            y: 0.5 * (self.y + other.y),
            z: 0.5 * (self.z + other.z),
        }
    }
}

/// An oriented edge `[src, dst]` with `src != dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrientedEdge {
    /// Tail (reference orientation).
    pub src: VertexId,
    /// Head (reference orientation).
    pub dst: VertexId,
}

/// An oriented triangle `[a, b, c]` with increasing vertex labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrientedFace {
    /// Three vertices in increasing-id order.
    pub verts: [VertexId; 3],
}

/// Simplicial 2-complex: downward-closed set of vertices, edges, and triangles.
///
/// Reference orientations follow increasing vertex labels (Yang et al., AISTATS
/// 2024, §2.2; Lim, SIAM Review 2020).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimplicialComplex2 {
    vertices: Vec<Point>,
    edges: Vec<OrientedEdge>,
    faces: Vec<OrientedFace>,
    /// Canonical unordered pair → edge id (the stored orientation is `src < dst`
    /// after construction; lookup still works if the caller swaps endpoints).
    #[serde(skip)]
    edge_of: HashMap<(usize, usize), EdgeId>,
}

impl SimplicialComplex2 {
    /// Build a complex, adding any missing face-boundary edges.
    pub fn new(
        vertices: Vec<Point>,
        edges: Vec<(usize, usize)>,
        faces: Vec<[usize; 3]>,
    ) -> Result<Self> {
        let n0 = vertices.len();
        let mut edge_set: HashMap<(usize, usize), ()> = HashMap::new();
        for (a, b) in edges {
            let (lo, hi) = ordered_pair(a, b, n0)?;
            edge_set.insert((lo, hi), ());
        }
        for face in &faces {
            let vs = sorted_triple(face[0], face[1], face[2], n0)?;
            edge_set.insert((vs[0], vs[1]), ());
            edge_set.insert((vs[1], vs[2]), ());
            edge_set.insert((vs[0], vs[2]), ());
        }

        let mut edge_pairs: Vec<(usize, usize)> = edge_set.into_keys().collect();
        edge_pairs.sort_unstable();
        let mut edge_of = HashMap::with_capacity(edge_pairs.len());
        let mut oriented_edges = Vec::with_capacity(edge_pairs.len());
        for (i, (lo, hi)) in edge_pairs.into_iter().enumerate() {
            edge_of.insert((lo, hi), EdgeId(i));
            oriented_edges.push(OrientedEdge {
                src: VertexId(lo),
                dst: VertexId(hi),
            });
        }

        let mut oriented_faces = Vec::with_capacity(faces.len());
        let mut seen_faces = HashMap::new();
        for face in faces {
            let vs = sorted_triple(face[0], face[1], face[2], n0)?;
            if seen_faces.insert(vs, ()).is_some() {
                continue;
            }
            oriented_faces.push(OrientedFace {
                verts: [VertexId(vs[0]), VertexId(vs[1]), VertexId(vs[2])],
            });
        }

        Ok(Self {
            vertices,
            edges: oriented_edges,
            faces: oriented_faces,
            edge_of,
        })
    }

    /// Empty complex.
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            edge_of: HashMap::new(),
        }
    }

    /// Rebuild the edge lookup table after deserialization.
    pub fn reindex(&mut self) {
        self.edge_of.clear();
        for (i, e) in self.edges.iter().enumerate() {
            let (lo, hi) = if e.src.0 < e.dst.0 {
                (e.src.0, e.dst.0)
            } else {
                (e.dst.0, e.src.0)
            };
            self.edge_of.insert((lo, hi), EdgeId(i));
        }
    }

    /// Number of vertices `N₀`.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges `N₁`.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of triangles `N₂`.
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Vertex coordinates.
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    /// Oriented edges.
    pub fn edges(&self) -> &[OrientedEdge] {
        &self.edges
    }

    /// Oriented faces.
    pub fn faces(&self) -> &[OrientedFace] {
        &self.faces
    }

    /// Look up the edge `{i,j}` (order ignored).
    pub fn edge_id(&self, a: VertexId, b: VertexId) -> Result<EdgeId> {
        let (lo, hi) = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
        self.edge_of
            .get(&(lo, hi))
            .copied()
            .ok_or_else(|| HodgekerError::InvalidSimplex(format!("no edge {{{lo},{hi}}}")))
    }

    /// Sign of the stored orientation relative to `(src, dst)`: `+1` if stored
    /// as `src → dst`, `-1` if stored reversed.
    pub fn edge_sign(&self, src: VertexId, dst: VertexId) -> Result<f64> {
        let id = self.edge_id(src, dst)?;
        let e = self.edges[id.0];
        if e.src == src && e.dst == dst {
            Ok(1.0)
        } else if e.src == dst && e.dst == src {
            Ok(-1.0)
        } else {
            Err(HodgekerError::InvalidSimplex(
                "edge orientation bookkeeping is inconsistent".into(),
            ))
        }
    }

    /// Midpoint of an edge (for CSV / plots).
    pub fn edge_midpoint(&self, e: EdgeId) -> Point {
        let edge = self.edges[e.0];
        self.vertices[edge.src.0].midpoint(self.vertices[edge.dst.0])
    }

    /// Face centroid.
    pub fn face_centroid(&self, f: FaceId) -> Point {
        let vs = self.faces[f.0].verts;
        let a = self.vertices[vs[0].0];
        let b = self.vertices[vs[1].0];
        let c = self.vertices[vs[2].0];
        Point {
            x: (a.x + b.x + c.x) / 3.0,
            y: (a.y + b.y + c.y) / 3.0,
            z: (a.z + b.z + c.z) / 3.0,
        }
    }

    /// Axis-aligned triangulated grid in the unit square (or integer lattice).
    ///
    /// `nx`, `ny` are **vertex** counts. Each quad is split into two triangles
    /// by the diagonal `(i,j) — (i+1,j+1)`.
    pub fn grid(nx: usize, ny: usize, with_faces: bool) -> Result<Self> {
        if nx < 2 || ny < 2 {
            return Err(HodgekerError::InvalidSimplex(
                "grid needs at least 2 vertices along each axis".into(),
            ));
        }
        let mut vertices = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                vertices.push(Point::xy(i as f64, j as f64));
            }
        }
        let idx = |i: usize, j: usize| i + j * nx;
        let mut edges = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                if i + 1 < nx {
                    edges.push((idx(i, j), idx(i + 1, j)));
                }
                if j + 1 < ny {
                    edges.push((idx(i, j), idx(i, j + 1)));
                }
                if with_faces && i + 1 < nx && j + 1 < ny {
                    edges.push((idx(i, j), idx(i + 1, j + 1)));
                }
            }
        }
        let mut faces = Vec::new();
        if with_faces {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let a = idx(i, j);
                    let b = idx(i + 1, j);
                    let c = idx(i + 1, j + 1);
                    let d = idx(i, j + 1);
                    faces.push([a, b, c]);
                    faces.push([a, c, d]);
                }
            }
        }
        Self::new(vertices, edges, faces)
    }

    /// Grid with a rectangular hole (faces omitted) so `β₁ ≥ 1`.
    ///
    /// `hole` is `[i0, i1) × [j0, j1)` in cell coordinates.
    pub fn grid_with_hole(
        nx: usize,
        ny: usize,
        hole: (usize, usize, usize, usize),
    ) -> Result<Self> {
        let mut sc = Self::grid(nx, ny, true)?;
        let (i0, i1, j0, j1) = hole;
        sc.faces.retain(|f| {
            let a = sc.vertices[f.verts[0].0];
            let b = sc.vertices[f.verts[1].0];
            let d = sc.vertices[f.verts[2].0];
            let cx = (a.x + b.x + d.x) / 3.0;
            let cy = (a.y + b.y + d.y) / 3.0;
            let ci = cx.floor() as usize;
            let cj = cy.floor() as usize;
            !(ci >= i0 && ci < i1 && cj >= j0 && cj < j1)
        });
        Ok(sc)
    }

    /// Single oriented triangle (the smallest interesting `SC₂`).
    pub fn triangle() -> Self {
        Self::new(
            vec![
                Point::xy(0.0, 0.0),
                Point::xy(1.0, 0.0),
                Point::xy(0.5, 0.866),
            ],
            vec![(0, 1), (1, 2), (0, 2)],
            vec![[0, 1, 2]],
        )
        .expect("canonical triangle")
    }
}

fn ordered_pair(a: usize, b: usize, n0: usize) -> Result<(usize, usize)> {
    if a >= n0 || b >= n0 {
        return Err(HodgekerError::InvalidSimplex(format!(
            "vertex id out of range ({a},{b}) vs n0={n0}"
        )));
    }
    if a == b {
        return Err(HodgekerError::InvalidSimplex(format!("loop at vertex {a}")));
    }
    Ok(if a < b { (a, b) } else { (b, a) })
}

fn sorted_triple(a: usize, b: usize, c: usize, n0: usize) -> Result<[usize; 3]> {
    if a >= n0 || b >= n0 || c >= n0 {
        return Err(HodgekerError::InvalidSimplex(format!(
            "face vertex out of range ({a},{b},{c}) vs n0={n0}"
        )));
    }
    if a == b || b == c || a == c {
        return Err(HodgekerError::InvalidSimplex(
            "degenerate face with repeated vertices".into(),
        ));
    }
    let mut vs = [a, b, c];
    vs.sort_unstable();
    Ok(vs)
}

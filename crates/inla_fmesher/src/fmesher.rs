use std::collections::HashMap;
use std::fs;
use std::path::Path;

const GEOM_EPSILON: f64 = 1e-15;

#[derive(Debug, Clone, Copy)]
pub struct Vertex2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle(pub [usize; 3]);

#[derive(Debug, Clone)]
pub struct Mesh2D {
    pub vertices: Vec<Vertex2>,
    pub triangles: Vec<Triangle>,
    pub neighbors: Vec<[Option<usize>; 3]>,
    pub neighbor_edge: Vec<[Option<usize>; 3]>,
    pub vertex_to_triangle: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
pub struct BoundaryInput {
    pub vertices: Vec<Vertex2>,
    pub boundary_indices: Vec<usize>,
    pub boundary_segments: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRef {
    pub triangle: usize,
    pub edge: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PointLocation {
    Triangle {
        triangle: usize,
        barycentric: [f64; 3],
    },
    Edge {
        triangle: usize,
        edge: usize,
        barycentric: [f64; 3],
    },
    Vertex {
        triangle: usize,
        vertex: usize,
        barycentric: [f64; 3],
    },
    Outside,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathStep {
    pub triangle: usize,
    pub exited_edge: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathTrace {
    pub start_triangle: usize,
    pub end_triangle: Option<usize>,
    pub crossed_edges: Vec<EdgeRef>,
    pub terminal: PointLocation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTriplet {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl SparseTriplet {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    fn add(&mut self, row: usize, col: usize, value: f64) {
        self.entries.push((row, col, value));
    }

    fn coalesce(mut self) -> Self {
        self.entries
            .sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut merged: Vec<(usize, usize, f64)> = Vec::with_capacity(self.entries.len());
        for (r, c, v) in self.entries {
            if let Some(last) = merged.last_mut()
                && last.0 == r && last.1 == c {
                    last.2 += v;
                    continue;
                }
            merged.push((r, c, v));
        }
        self.entries = merged;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FemBlocks {
    pub c0: SparseTriplet,
    pub c1: SparseTriplet,
    pub g1: SparseTriplet,
    pub b1: SparseTriplet,
    pub triangle_areas: Vec<f64>,
}

pub fn build_mesh2d(vertices: Vec<Vertex2>, triangles: Vec<Triangle>) -> Result<Mesh2D, String> {
    validate_triangles(&vertices, &triangles)?;

    let mut neighbors = vec![[None; 3]; triangles.len()];
    let mut neighbor_edge = vec![[None; 3]; triangles.len()];
    let mut edge_map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

    for (t_idx, tri) in triangles.iter().enumerate() {
        for edge_idx in 0..3 {
            let a = tri.0[(edge_idx + 1) % 3];
            let b = tri.0[(edge_idx + 2) % 3];
            if let Some((other_t, other_edge)) = edge_map.remove(&(b, a)) {
                neighbors[t_idx][edge_idx] = Some(other_t);
                neighbor_edge[t_idx][edge_idx] = Some(other_edge);
                neighbors[other_t][other_edge] = Some(t_idx);
                neighbor_edge[other_t][other_edge] = Some(edge_idx);
            } else if edge_map.insert((a, b), (t_idx, edge_idx)).is_some() {
                return Err(format!(
                    "non-manifold edge detected between vertices {a} and {b}"
                ));
            }
        }
    }

    let mut vertex_to_triangle = vec![None; vertices.len()];
    for (t_idx, tri) in triangles.iter().enumerate() {
        for v in tri.0 {
            if vertex_to_triangle[v].is_none() {
                vertex_to_triangle[v] = Some(t_idx);
            }
        }
    }

    Ok(Mesh2D {
        vertices,
        triangles,
        neighbors,
        neighbor_edge,
        vertex_to_triangle,
    })
}

impl Mesh2D {
    pub fn locate_point(&self, point: Vertex2) -> PointLocation {
        if self.triangles.is_empty() {
            return PointLocation::Outside;
        }
        self.locate_point_from(0, point)
    }

    /// Build piecewise-linear FEM projector rows for observation locations.
    ///
    /// For each point inside (or on the boundary of) a triangle, returns up to
    /// three triplets `(obs_row, vertex_col, barycentric_weight)`. Points that
    /// fall outside the mesh return an error (R-INLA would require a domain
    /// extension / nearest-boundary policy).
    pub fn observation_projector_triplets(
        &self,
        points: &[Vertex2],
    ) -> Result<Vec<(usize, usize, f64)>, String> {
        let mut trips = Vec::with_capacity(points.len() * 3);
        let mut start = 0usize;
        for (row, &p) in points.iter().enumerate() {
            let loc = if self.triangles.is_empty() {
                PointLocation::Outside
            } else {
                self.locate_point_from(start, p)
            };
            let (tri_idx, bary) = match loc {
                PointLocation::Triangle {
                    triangle,
                    barycentric,
                }
                | PointLocation::Edge {
                    triangle,
                    barycentric,
                    ..
                }
                | PointLocation::Vertex {
                    triangle,
                    barycentric,
                    ..
                } => (triangle, barycentric),
                PointLocation::Outside => {
                    return Err(format!(
                        "observation point ({}, {}) is outside the mesh",
                        p.x, p.y
                    ));
                }
            };
            start = tri_idx;
            let verts = self.triangles[tri_idx].0;
            for k in 0..3 {
                let w = bary[k];
                if w.abs() > GEOM_EPSILON {
                    trips.push((row, verts[k], w));
                }
            }
        }
        Ok(trips)
    }

    pub fn locate_point_from(&self, start_triangle: usize, point: Vertex2) -> PointLocation {
        if self.triangles.is_empty() || start_triangle >= self.triangles.len() {
            return PointLocation::Outside;
        }

        let mut current = start_triangle;
        let mut remaining = self.triangles.len().saturating_mul(3).max(1);

        while remaining > 0 {
            remaining -= 1;
            let tri = self.triangles[current].0;
            let orientation = signed_area2(
                self.vertices[tri[0]],
                self.vertices[tri[1]],
                self.vertices[tri[2]],
            );
            let orient_sign = if orientation >= 0.0 { 1.0 } else { -1.0 };

            let mut most_negative = 0.0;
            let mut leaving_edge = None;
            for edge in 0..3 {
                let a = self.vertices[tri[(edge + 1) % 3]];
                let b = self.vertices[tri[(edge + 2) % 3]];
                let side = orient_sign * signed_area2(a, b, point);
                if side < most_negative {
                    most_negative = side;
                    leaving_edge = Some(edge);
                }
            }

            if most_negative >= -GEOM_EPSILON {
                return classify_point_in_triangle(current, tri, point, &self.vertices);
            }

            let edge = match leaving_edge {
                Some(e) => e,
                None => return PointLocation::Outside,
            };
            match self.neighbors[current][edge] {
                Some(next) => current = next,
                None => return PointLocation::Outside,
            }
        }

        PointLocation::Outside
    }

    pub fn trace_path(&self, start: Vertex2, end: Vertex2) -> PathTrace {
        if self.triangles.is_empty() {
            return PathTrace {
                start_triangle: 0,
                end_triangle: None,
                crossed_edges: Vec::new(),
                terminal: PointLocation::Outside,
            };
        }

        let start_loc = self.locate_point(start);
        let start_triangle = match &start_loc {
            PointLocation::Triangle { triangle, .. }
            | PointLocation::Edge { triangle, .. }
            | PointLocation::Vertex { triangle, .. } => *triangle,
            PointLocation::Outside => {
                return PathTrace {
                    start_triangle: 0,
                    end_triangle: None,
                    crossed_edges: Vec::new(),
                    terminal: PointLocation::Outside,
                };
            }
        };

        self.trace_path_from_triangle(start_triangle, start, end)
    }

    pub fn trace_path_from_triangle(
        &self,
        start_triangle: usize,
        _start: Vertex2,
        end: Vertex2,
    ) -> PathTrace {
        if self.triangles.is_empty() || start_triangle >= self.triangles.len() {
            return PathTrace {
                start_triangle,
                end_triangle: None,
                crossed_edges: Vec::new(),
                terminal: PointLocation::Outside,
            };
        }

        let mut current = start_triangle;
        let mut crossed_edges = Vec::new();
        let mut remaining = self.triangles.len().saturating_mul(3).max(1);

        while remaining > 0 {
            remaining -= 1;
            let tri = self.triangles[current].0;
            let orientation = signed_area2(
                self.vertices[tri[0]],
                self.vertices[tri[1]],
                self.vertices[tri[2]],
            );
            let orient_sign = if orientation >= 0.0 { 1.0 } else { -1.0 };

            let end_bary = barycentric_coords(
                self.vertices[tri[0]],
                self.vertices[tri[1]],
                self.vertices[tri[2]],
                end,
            );
            if end_bary.iter().all(|w| *w >= -GEOM_EPSILON) {
                return PathTrace {
                    start_triangle,
                    end_triangle: Some(current),
                    crossed_edges,
                    terminal: classify_point_in_triangle(current, tri, end, &self.vertices),
                };
            }

            let mut most_negative = 0.0;
            let mut leaving_edge = None;
            for edge in 0..3 {
                let a = self.vertices[tri[(edge + 1) % 3]];
                let b = self.vertices[tri[(edge + 2) % 3]];
                let side = orient_sign * signed_area2(a, b, end);
                if side < most_negative {
                    most_negative = side;
                    leaving_edge = Some(edge);
                }
            }

            let edge = match leaving_edge {
                Some(e) => e,
                None => {
                    return PathTrace {
                        start_triangle,
                        end_triangle: Some(current),
                        crossed_edges,
                        terminal: classify_point_in_triangle(current, tri, end, &self.vertices),
                    };
                }
            };

            crossed_edges.push(EdgeRef {
                triangle: current,
                edge,
            });

            match self.neighbors[current][edge] {
                Some(next) => current = next,
                None => {
                    return PathTrace {
                        start_triangle,
                        end_triangle: None,
                        crossed_edges,
                        terminal: PointLocation::Outside,
                    };
                }
            }
        }

        PathTrace {
            start_triangle,
            end_triangle: None,
            crossed_edges,
            terminal: PointLocation::Outside,
        }
    }

    pub fn split_edge(&self, edge_ref: EdgeRef, new_vertex: Vertex2) -> Result<Mesh2D, String> {
        let (t0, e0) = self.validate_edge_ref(edge_ref)?;
        let tri0 = self.triangles[t0].0;

        let v0 = tri0[(e0 + 1) % 3];
        let v1 = tri0[(e0 + 2) % 3];
        let n0 = tri0[e0];

        let mut new_vertices = self.vertices.clone();
        let new_idx = new_vertices.len();
        new_vertices.push(new_vertex);

        let mut affected = vec![t0];
        if let Some(t1) = self.neighbors[t0][e0] {
            affected.push(t1);
        }

        let mut new_triangles = Vec::with_capacity(self.triangles.len() + 2);
        for (idx, tri) in self.triangles.iter().enumerate() {
            if !affected.contains(&idx) {
                new_triangles.push(*tri);
            }
        }

        new_triangles.push(oriented_triangle(
            &new_vertices,
            self.triangles[t0],
            [n0, v0, new_idx],
        ));
        new_triangles.push(oriented_triangle(
            &new_vertices,
            self.triangles[t0],
            [n0, new_idx, v1],
        ));

        if let Some(t1) = self.neighbors[t0][e0] {
            let e1 = triangle_edge_index(self.triangles[t1], v0, v1).ok_or_else(|| {
                format!(
                    "inconsistent topology: edge ({v0}, {v1}) not found in neighbor triangle {t1}"
                )
            })?;
            let tri1 = self.triangles[t1].0;
            let n1 = tri1[e1];
            new_triangles.push(oriented_triangle(
                &new_vertices,
                self.triangles[t1],
                [n1, tri1[(e1 + 1) % 3], new_idx],
            ));
            new_triangles.push(oriented_triangle(
                &new_vertices,
                self.triangles[t1],
                [n1, new_idx, tri1[(e1 + 2) % 3]],
            ));
        }

        build_mesh2d(new_vertices, new_triangles)
    }

    pub fn swap_edge(&self, edge_ref: EdgeRef) -> Result<Mesh2D, String> {
        let (t0, e0) = self.validate_edge_ref(edge_ref)?;
        let t1 = self.neighbors[t0][e0].ok_or_else(|| "cannot swap boundary edge".to_string())?;

        let tri0 = self.triangles[t0].0;
        let a = tri0[(e0 + 1) % 3];
        let b = tri0[(e0 + 2) % 3];
        let c = tri0[e0];

        let e1 = triangle_edge_index(self.triangles[t1], a, b).ok_or_else(|| {
            format!("inconsistent topology: edge ({a}, {b}) not found in triangle {t1}")
        })?;
        let tri1 = self.triangles[t1].0;
        let d = tri1[e1];

        if !is_swapable_quad(
            self.vertices[a],
            self.vertices[b],
            self.vertices[c],
            self.vertices[d],
        ) {
            return Err("edge is not swapable (non-convex or degenerate local quad)".to_string());
        }

        let mut new_triangles = Vec::with_capacity(self.triangles.len());
        for (idx, tri) in self.triangles.iter().enumerate() {
            if idx != t0 && idx != t1 {
                new_triangles.push(*tri);
            }
        }

        new_triangles.push(oriented_triangle(
            &self.vertices,
            self.triangles[t0],
            [a, d, c],
        ));
        new_triangles.push(oriented_triangle(
            &self.vertices,
            self.triangles[t1],
            [b, c, d],
        ));

        build_mesh2d(self.vertices.clone(), new_triangles)
    }

    pub fn assemble_fem_blocks(&self) -> FemBlocks {
        let mut c0 = SparseTriplet::new(self.vertices.len(), self.vertices.len());
        let mut c1 = SparseTriplet::new(self.vertices.len(), self.vertices.len());
        let mut g1 = SparseTriplet::new(self.vertices.len(), self.vertices.len());
        let mut b1 = SparseTriplet::new(self.vertices.len(), self.vertices.len());
        let mut triangle_areas = Vec::with_capacity(self.triangles.len());

        for (t, tri) in self.triangles.iter().enumerate() {
            let tv = tri.0;
            let s0 = self.vertices[tv[0]];
            let s1 = self.vertices[tv[1]];
            let s2 = self.vertices[tv[2]];

            let e0 = edge_vec(s2, s1);
            let e1 = edge_vec(s0, s2);
            let e2 = edge_vec(s1, s0);
            let e = [e0, e1, e2];

            let area = 0.5 * signed_area2(s0, s1, s2).abs();
            triangle_areas.push(area);
            let fa = 0.5 * cross2(e0, e1).abs();
            let fa = if fa <= GEOM_EPSILON { area } else { fa };

            let mut eij = [[0.0; 3]; 3];
            for i in 0..3 {
                eij[i][i] = dot(e[i], e[i]);
                for j in (i + 1)..3 {
                    eij[i][j] = dot(e[i], e[j]);
                    eij[j][i] = eij[i][j];
                }
            }

            for i in 0..3 {
                c0.add(tv[i], tv[i], area / 3.0);
                c1.add(tv[i], tv[i], area / 6.0);
                g1.add(tv[i], tv[i], eij[i][i] / (4.0 * fa));
                for j in (i + 1)..3 {
                    c1.add(tv[i], tv[j], area / 12.0);
                    c1.add(tv[j], tv[i], area / 12.0);
                    let vij = eij[i][j] / (4.0 * fa);
                    g1.add(tv[i], tv[j], vij);
                    g1.add(tv[j], tv[i], vij);
                }
            }

            let boundary = [
                self.neighbors[t][0].is_none(),
                self.neighbors[t][1].is_none(),
                self.neighbors[t][2].is_none(),
            ];
            if boundary.iter().any(|b| *b) {
                let vij = -1.0 / (4.0 * fa);
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            if boundary[k] && i != k {
                                b1.add(tv[i], tv[j], eij[k][j] * vij);
                            }
                        }
                    }
                }
            }
        }

        FemBlocks {
            c0: c0.coalesce(),
            c1: c1.coalesce(),
            g1: g1.coalesce(),
            b1: b1.coalesce(),
            triangle_areas,
        }
    }

    fn validate_edge_ref(&self, edge_ref: EdgeRef) -> Result<(usize, usize), String> {
        if edge_ref.triangle >= self.triangles.len() {
            return Err(format!("triangle index {} out of range", edge_ref.triangle));
        }
        if edge_ref.edge >= 3 {
            return Err(format!("edge index {} out of range [0, 2]", edge_ref.edge));
        }
        Ok((edge_ref.triangle, edge_ref.edge))
    }
}

pub fn read_positions_xy<P: AsRef<Path>>(path: P) -> Result<Vec<Vertex2>, String> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read positions file '{}': {e}",
            path.as_ref().display()
        )
    })?;

    let mut vertices = Vec::new();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let x = match parts.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        let y = match parts.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        vertices.push(Vertex2 { x, y });
    }

    if vertices.is_empty() {
        return Err("positions file did not contain any parseable 'x y' rows".to_string());
    }
    Ok(vertices)
}

pub fn read_boundary_indices<P: AsRef<Path>>(path: P) -> Result<Vec<usize>, String> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read boundary file '{}': {e}",
            path.as_ref().display()
        )
    })?;

    let mut raw_indices = Vec::new();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Ok(idx) = line.parse::<usize>() {
            raw_indices.push(idx);
        }
    }

    if raw_indices.len() < 2 {
        return Err("boundary file must contain at least two vertex indices".to_string());
    }

    Ok(raw_indices)
}

pub fn build_boundary_segments(indices: &[usize]) -> Vec<[usize; 2]> {
    if indices.len() < 2 {
        return Vec::new();
    }

    let mut segments = Vec::with_capacity(indices.len().saturating_sub(1));
    for pair in indices.windows(2) {
        if pair[0] != pair[1] {
            segments.push([pair[0], pair[1]]);
        }
    }
    segments
}

pub fn load_fmesher_boundary_input<P1: AsRef<Path>, P2: AsRef<Path>>(
    positions_path: P1,
    boundary_path: P2,
) -> Result<BoundaryInput, String> {
    let vertices = read_positions_xy(&positions_path)?;
    let raw_boundary = read_boundary_indices(&boundary_path)?;
    let boundary_indices = normalize_boundary_indices(&raw_boundary, vertices.len())?;

    for idx in &boundary_indices {
        if *idx >= vertices.len() {
            return Err(format!(
                "boundary index {idx} out of range for {} vertices",
                vertices.len()
            ));
        }
    }

    let boundary_segments = build_boundary_segments(&boundary_indices);
    Ok(BoundaryInput {
        vertices,
        boundary_indices,
        boundary_segments,
    })
}

pub fn load_fmesher_raw_boundary_input<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
    positions_path: P1,
    boundary_points_path: P2,
    boundary_index_path: P3,
) -> Result<BoundaryInput, String> {
    let mut vertices = read_positions_xy(&positions_path)?;
    let boundary_vertices = read_positions_xy(&boundary_points_path)?;
    vertices.extend(boundary_vertices);

    let raw_boundary = read_boundary_indices(&boundary_index_path)?;
    let boundary_indices = normalize_boundary_indices(&raw_boundary, vertices.len())?;
    for idx in &boundary_indices {
        if *idx >= vertices.len() {
            return Err(format!(
                "boundary index {idx} out of range for {} vertices",
                vertices.len()
            ));
        }
    }

    let boundary_segments = build_boundary_segments(&boundary_indices);
    Ok(BoundaryInput {
        vertices,
        boundary_indices,
        boundary_segments,
    })
}

fn normalize_boundary_indices(indices: &[usize], n_vertices: usize) -> Result<Vec<usize>, String> {
    if indices.iter().all(|idx| *idx < n_vertices) {
        return Ok(indices.to_vec());
    }

    if indices.iter().all(|idx| *idx >= 1 && *idx <= n_vertices) {
        return Ok(indices.iter().map(|idx| idx - 1).collect());
    }

    let max_idx = indices.iter().copied().max().unwrap_or(0);
    Err(format!(
        "boundary indices are incompatible with vertex count: max index {max_idx}, vertices {n_vertices}"
    ))
}

fn classify_point_in_triangle(
    triangle_idx: usize,
    tri: [usize; 3],
    point: Vertex2,
    vertices: &[Vertex2],
) -> PointLocation {
    let bary = barycentric_coords(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]], point);

    let near_zero = [
        bary[0].abs() <= GEOM_EPSILON,
        bary[1].abs() <= GEOM_EPSILON,
        bary[2].abs() <= GEOM_EPSILON,
    ];

    for i in 0..3 {
        if near_zero[(i + 1) % 3] && near_zero[(i + 2) % 3] {
            return PointLocation::Vertex {
                triangle: triangle_idx,
                vertex: tri[i],
                barycentric: bary,
            };
        }
    }

    for edge in 0..3 {
        if near_zero[edge] {
            return PointLocation::Edge {
                triangle: triangle_idx,
                edge,
                barycentric: bary,
            };
        }
    }

    PointLocation::Triangle {
        triangle: triangle_idx,
        barycentric: bary,
    }
}

fn barycentric_coords(a: Vertex2, b: Vertex2, c: Vertex2, p: Vertex2) -> [f64; 3] {
    let denom = signed_area2(a, b, c);
    if denom.abs() <= GEOM_EPSILON {
        return [f64::NAN, f64::NAN, f64::NAN];
    }
    let w0 = signed_area2(b, c, p) / denom;
    let w1 = signed_area2(c, a, p) / denom;
    let w2 = 1.0 - w0 - w1;
    [w0, w1, w2]
}

fn triangle_edge_index(triangle: Triangle, u: usize, v: usize) -> Option<usize> {
    for edge in 0..3 {
        let a = triangle.0[(edge + 1) % 3];
        let b = triangle.0[(edge + 2) % 3];
        if (a == u && b == v) || (a == v && b == u) {
            return Some(edge);
        }
    }
    None
}

fn oriented_triangle(vertices: &[Vertex2], reference: Triangle, candidate: [usize; 3]) -> Triangle {
    let ref_sign = signed_area2(
        vertices[reference.0[0]],
        vertices[reference.0[1]],
        vertices[reference.0[2]],
    );
    let cand_sign = signed_area2(
        vertices[candidate[0]],
        vertices[candidate[1]],
        vertices[candidate[2]],
    );
    if ref_sign * cand_sign < 0.0 {
        Triangle([candidate[0], candidate[2], candidate[1]])
    } else {
        Triangle(candidate)
    }
}

fn is_swapable_quad(a: Vertex2, b: Vertex2, c: Vertex2, d: Vertex2) -> bool {
    let ab_c = signed_area2(a, b, c);
    let ab_d = signed_area2(a, b, d);
    let cd_a = signed_area2(c, d, a);
    let cd_b = signed_area2(c, d, b);
    (ab_c * ab_d) < -GEOM_EPSILON && (cd_a * cd_b) < -GEOM_EPSILON
}

fn validate_triangles(vertices: &[Vertex2], triangles: &[Triangle]) -> Result<(), String> {
    if vertices.len() < 3 {
        return Err("need at least three vertices".to_string());
    }

    for (t_idx, tri) in triangles.iter().enumerate() {
        let [a, b, c] = tri.0;
        if a >= vertices.len() || b >= vertices.len() || c >= vertices.len() {
            return Err(format!(
                "triangle {t_idx} has vertex out of range: [{a}, {b}, {c}] with {} vertices",
                vertices.len()
            ));
        }
        if a == b || b == c || a == c {
            return Err(format!(
                "triangle {t_idx} is degenerate (duplicate indices)"
            ));
        }

        let area2 = signed_area2(vertices[a], vertices[b], vertices[c]).abs();
        if area2 <= GEOM_EPSILON {
            return Err(format!("triangle {t_idx} has near-zero area"));
        }
    }
    Ok(())
}

fn signed_area2(a: Vertex2, b: Vertex2, c: Vertex2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn edge_vec(a: Vertex2, b: Vertex2) -> Vertex2 {
    Vertex2 {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

fn dot(a: Vertex2, b: Vertex2) -> f64 {
    a.x * b.x + a.y * b.y
}

fn cross2(a: Vertex2, b: Vertex2) -> f64 {
    a.x * b.y - a.y * b.x
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn builds_topology_for_two_triangles() {
        let vertices = vec![
            Vertex2 { x: 0.0, y: 0.0 },
            Vertex2 { x: 1.0, y: 0.0 },
            Vertex2 { x: 0.0, y: 1.0 },
            Vertex2 { x: 1.0, y: 1.0 },
        ];
        let triangles = vec![Triangle([0, 1, 2]), Triangle([1, 3, 2])];
        let mesh = build_mesh2d(vertices, triangles).expect("mesh should build");

        assert_eq!(mesh.neighbors[0][0], Some(1));
        assert_eq!(mesh.neighbors[1][1], Some(0));
        assert!(mesh.vertex_to_triangle.iter().all(|v| v.is_some()));
    }

    #[test]
    fn builds_boundary_segments() {
        let seg = build_boundary_segments(&[5, 6, 8, 8, 9]);
        assert_eq!(seg, vec![[5, 6], [6, 8], [8, 9]]);
    }

    #[test]
    fn loads_koala_boundary_input() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let positions = root.join("../../fmesher/examples/koala-positions.txt");
        let boundary_points = root.join("../../fmesher/examples/koala-boundary.txt");
        let boundary = root.join("../../fmesher/examples/koala-bnd0.txt");

        let out = load_fmesher_raw_boundary_input(positions, boundary_points, boundary)
            .expect("load koala inputs");
        assert!(out.vertices.len() > 900);
        assert!(out.boundary_indices.len() > 20);
        assert_eq!(out.boundary_segments.len(), out.boundary_indices.len() - 1);
    }

    #[test]
    fn swaps_internal_edge() {
        let mesh = build_mesh2d(
            vec![
                Vertex2 { x: 0.0, y: 0.0 },
                Vertex2 { x: 1.0, y: 0.0 },
                Vertex2 { x: 0.0, y: 1.0 },
                Vertex2 { x: 1.0, y: 1.0 },
            ],
            vec![Triangle([0, 1, 2]), Triangle([1, 3, 2])],
        )
        .expect("mesh should build");

        let swapped = mesh
            .swap_edge(EdgeRef {
                triangle: 0,
                edge: 0,
            })
            .expect("swap should succeed");

        assert_eq!(swapped.triangles.len(), 2);
        assert!(
            triangle_edge_index(swapped.triangles[0], 0, 3).is_some()
                || triangle_edge_index(swapped.triangles[1], 0, 3).is_some()
        );
        assert!(
            triangle_edge_index(swapped.triangles[0], 1, 2).is_none()
                && triangle_edge_index(swapped.triangles[1], 1, 2).is_none()
        );
    }

    #[test]
    fn splits_internal_edge() {
        let mesh = build_mesh2d(
            vec![
                Vertex2 { x: 0.0, y: 0.0 },
                Vertex2 { x: 1.0, y: 0.0 },
                Vertex2 { x: 0.0, y: 1.0 },
                Vertex2 { x: 1.0, y: 1.0 },
            ],
            vec![Triangle([0, 1, 2]), Triangle([1, 3, 2])],
        )
        .expect("mesh should build");

        let split = mesh
            .split_edge(
                EdgeRef {
                    triangle: 0,
                    edge: 0,
                },
                Vertex2 { x: 0.5, y: 0.5 },
            )
            .expect("split should succeed");

        assert_eq!(split.vertices.len(), 5);
        assert_eq!(split.triangles.len(), 4);
    }

    #[test]
    fn splits_boundary_edge() {
        let mesh = build_mesh2d(
            vec![
                Vertex2 { x: 0.0, y: 0.0 },
                Vertex2 { x: 1.0, y: 0.0 },
                Vertex2 { x: 0.0, y: 1.0 },
            ],
            vec![Triangle([0, 1, 2])],
        )
        .expect("mesh should build");

        let split = mesh
            .split_edge(
                EdgeRef {
                    triangle: 0,
                    edge: 0,
                },
                Vertex2 { x: 0.5, y: 0.5 },
            )
            .expect("boundary split should succeed");

        assert_eq!(split.vertices.len(), 4);
        assert_eq!(split.triangles.len(), 2);
    }

    #[test]
    fn locates_point_in_triangle_edge_and_vertex() {
        let mesh = build_mesh2d(
            vec![
                Vertex2 { x: 0.0, y: 0.0 },
                Vertex2 { x: 1.0, y: 0.0 },
                Vertex2 { x: 0.0, y: 1.0 },
                Vertex2 { x: 1.0, y: 1.0 },
            ],
            vec![Triangle([0, 1, 2]), Triangle([1, 3, 2])],
        )
        .expect("mesh should build");

        match mesh.locate_point(Vertex2 { x: 0.1, y: 0.1 }) {
            PointLocation::Triangle { .. } => {}
            other => panic!("expected Triangle, got {other:?}"),
        }
        match mesh.locate_point(Vertex2 { x: 0.5, y: 0.5 }) {
            PointLocation::Edge { .. } | PointLocation::Vertex { .. } => {}
            other => panic!("expected Edge/Vertex, got {other:?}"),
        }
        match mesh.locate_point(Vertex2 { x: 0.0, y: 0.0 }) {
            PointLocation::Vertex { vertex, .. } => assert_eq!(vertex, 0),
            other => panic!("expected Vertex, got {other:?}"),
        }
        assert_eq!(
            mesh.locate_point(Vertex2 { x: 2.0, y: 2.0 }),
            PointLocation::Outside
        );
    }

    #[test]
    fn traces_path_across_internal_edge() {
        let mesh = build_mesh2d(
            vec![
                Vertex2 { x: 0.0, y: 0.0 },
                Vertex2 { x: 1.0, y: 0.0 },
                Vertex2 { x: 0.0, y: 1.0 },
                Vertex2 { x: 1.0, y: 1.0 },
            ],
            vec![Triangle([0, 1, 2]), Triangle([1, 3, 2])],
        )
        .expect("mesh should build");

        let trace = mesh.trace_path(Vertex2 { x: 0.1, y: 0.1 }, Vertex2 { x: 0.9, y: 0.9 });
        assert_eq!(trace.start_triangle, 0);
        assert_eq!(trace.end_triangle, Some(1));
        assert_eq!(trace.crossed_edges.len(), 1);
        assert_eq!(
            trace.crossed_edges[0],
            EdgeRef {
                triangle: 0,
                edge: 0
            }
        );
    }

    #[test]
    fn assembles_fem_blocks_for_square_mesh() {
        let mesh = build_mesh2d(
            vec![
                Vertex2 { x: 0.0, y: 0.0 },
                Vertex2 { x: 1.0, y: 0.0 },
                Vertex2 { x: 0.0, y: 1.0 },
                Vertex2 { x: 1.0, y: 1.0 },
            ],
            vec![Triangle([0, 1, 2]), Triangle([1, 3, 2])],
        )
        .expect("mesh should build");

        let blocks = mesh.assemble_fem_blocks();
        assert_eq!(blocks.triangle_areas, vec![0.5, 0.5]);
        assert_eq!(sum_entry(&blocks.c0, 0, 0), 1.0 / 6.0);
        assert_eq!(sum_entry(&blocks.c0, 1, 1), 1.0 / 3.0);
        assert_eq!(sum_entry(&blocks.c0, 2, 2), 1.0 / 3.0);
        assert_eq!(sum_entry(&blocks.c0, 3, 3), 1.0 / 6.0);
        assert!(!blocks.g1.entries.is_empty());
        assert!(!blocks.b1.entries.is_empty());
    }

    fn sum_entry(matrix: &SparseTriplet, row: usize, col: usize) -> f64 {
        matrix
            .entries
            .iter()
            .filter(|(r, c, _)| *r == row && *c == col)
            .map(|(_, _, v)| *v)
            .sum()
    }
}

//! 1D FEM mesh (linear elements on an ordered knot sequence).
//!
//! Mass / stiffness use the same lumped-C / gradient-G convention as the 2D
//! assembler so [`crate::FemBlocks`] can feed the existing SPDE `Q(κ, τ)`.

use crate::fmesher::{FemBlocks, SparseTriplet};

/// Knots on a line, strictly increasing.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh1D {
    pub loc: Vec<f64>,
}

/// Build a 1D mesh from knot locations. Duplicates and unsorted input are rejected.
pub fn build_mesh1d(loc: Vec<f64>) -> Result<Mesh1D, String> {
    if loc.len() < 2 {
        return Err("1D mesh requires at least two knots".into());
    }
    for (i, &x) in loc.iter().enumerate() {
        if !x.is_finite() {
            return Err(format!("1D mesh knot {i} is not finite"));
        }
        if i > 0 && loc[i] <= loc[i - 1] {
            return Err(format!(
                "1D mesh knots must be strictly increasing: loc[{}]={} <= loc[{}]={}",
                i,
                loc[i],
                i - 1,
                loc[i - 1]
            ));
        }
    }
    Ok(Mesh1D { loc })
}

impl Mesh1D {
    pub fn n(&self) -> usize {
        self.loc.len()
    }

    /// Lumped mass `c0`, consistent mass `c1`, and stiffness `g1`.
    pub fn assemble_fem_blocks(&self) -> FemBlocks {
        let n = self.n();
        let mut c0 = SparseTriplet::new(n, n);
        let mut c1 = SparseTriplet::new(n, n);
        let mut g1 = SparseTriplet::new(n, n);
        let b1 = SparseTriplet::new(n, n);
        let mut interval_lengths = Vec::with_capacity(n.saturating_sub(1));

        for i in 0..n - 1 {
            let h = self.loc[i + 1] - self.loc[i];
            interval_lengths.push(h);
            c0.add(i, i, h / 2.0);
            c0.add(i + 1, i + 1, h / 2.0);

            c1.add(i, i, h / 3.0);
            c1.add(i + 1, i + 1, h / 3.0);
            c1.add(i, i + 1, h / 6.0);
            c1.add(i + 1, i, h / 6.0);

            let inv_h = 1.0 / h;
            g1.add(i, i, inv_h);
            g1.add(i + 1, i + 1, inv_h);
            g1.add(i, i + 1, -inv_h);
            g1.add(i + 1, i, -inv_h);
        }

        FemBlocks {
            c0: c0.coalesce(),
            c1: c1.coalesce(),
            g1: g1.coalesce(),
            b1: b1.coalesce(),
            triangle_areas: interval_lengths,
        }
    }

    /// Piecewise-linear projector rows: `η(s) = (1-w) x_i + w x_{i+1}`.
    ///
    /// Locations outside `[loc[0], loc[n-1]]` clamp to the nearest endpoint
    /// (R-INLA `inla.mesh.1d` default for evaluation on a finite interval).
    pub fn observation_projector_triplets(
        &self,
        points: &[f64],
    ) -> Result<Vec<(usize, usize, f64)>, String> {
        let n = self.n();
        if n == 0 {
            return Err("1D projector requires a non-empty mesh".into());
        }
        let mut trips = Vec::with_capacity(points.len() * 2);
        for (row, &s) in points.iter().enumerate() {
            if !s.is_finite() {
                return Err(format!("1D observation location {row} is not finite"));
            }
            let (i, w) = self.interp_weight(s);
            if w.abs() <= 1e-15 {
                trips.push((row, i, 1.0));
            } else if (w - 1.0).abs() <= 1e-15 {
                trips.push((row, i + 1, 1.0));
            } else {
                trips.push((row, i, 1.0 - w));
                trips.push((row, i + 1, w));
            }
        }
        Ok(trips)
    }

    fn interp_weight(&self, s: f64) -> (usize, f64) {
        let n = self.n();
        let left = self.loc[0];
        let right = self.loc[n - 1];
        if s <= left {
            return (0, 0.0);
        }
        if s >= right {
            return (n - 2, 1.0);
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.loc[mid] <= s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let h = self.loc[lo + 1] - self.loc[lo];
        (lo, (s - self.loc[lo]) / h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(blocks: &FemBlocks, which: &str, r: usize, c: usize) -> f64 {
        let m = match which {
            "c0" => &blocks.c0,
            "c1" => &blocks.c1,
            "g1" => &blocks.g1,
            _ => panic!("bad"),
        };
        m.entries
            .iter()
            .filter(|(i, j, _)| *i == r && *j == c)
            .map(|(_, _, v)| *v)
            .sum()
    }

    #[test]
    fn unit_interval_two_knots() {
        let mesh = build_mesh1d(vec![0.0, 1.0]).unwrap();
        let fem = mesh.assemble_fem_blocks();
        assert_eq!(entry(&fem, "c0", 0, 0), 0.5);
        assert_eq!(entry(&fem, "c0", 1, 1), 0.5);
        assert!((entry(&fem, "g1", 0, 0) - 1.0).abs() < 1e-14);
        assert!((entry(&fem, "g1", 0, 1) + 1.0).abs() < 1e-14);
        assert_eq!(fem.triangle_areas, vec![1.0]);
    }

    #[test]
    fn projector_at_knots_and_midpoint() {
        let mesh = build_mesh1d(vec![0.0, 2.0, 5.0]).unwrap();
        let trips = mesh
            .observation_projector_triplets(&[0.0, 2.0, 3.5, 5.0, -1.0, 9.0])
            .unwrap();
        let mut a = vec![vec![0.0; 3]; 6];
        for (r, c, v) in trips {
            a[r][c] += v;
        }
        assert!((a[0][0] - 1.0).abs() < 1e-12);
        assert!((a[1][1] - 1.0).abs() < 1e-12);
        assert!((a[2][1] - 0.5).abs() < 1e-12);
        assert!((a[2][2] - 0.5).abs() < 1e-12);
        assert!((a[3][2] - 1.0).abs() < 1e-12);
        assert!((a[4][0] - 1.0).abs() < 1e-12);
        assert!((a[5][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_unsorted() {
        assert!(build_mesh1d(vec![1.0, 0.0]).is_err());
        assert!(build_mesh1d(vec![0.0, 0.0]).is_err());
    }
}

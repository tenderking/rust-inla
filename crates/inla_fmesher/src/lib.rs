//! Geometry and mesh topology (legacy `fmesher`).

mod fmesher;
mod mesh;
mod mesh1d;

pub use fmesher::*;
pub use mesh::{MeshSummary, read_mesh_summary};
pub use mesh1d::{Mesh1D, build_mesh1d};

//! Geometry and mesh topology (legacy `fmesher`).

mod fmesher;
mod mesh;

pub use fmesher::*;
pub use mesh::{MeshSummary, read_mesh_summary};

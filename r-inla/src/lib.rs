//! R bindings for rust-inla (`extendr`).
//!
//! Implementation lives in focused modules; this file only wires exports.

mod convert;
mod inference;
mod mesh;
mod precision;
mod priors;
mod registry;

use extendr_api::prelude::*;

extendr_module! {
    mod inla_rs;
    use precision;
    use mesh;
    use inference;
    use priors;
    use registry;
}

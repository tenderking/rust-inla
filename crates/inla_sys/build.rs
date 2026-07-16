use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/gmrflib_wrapper.h");
    println!("cargo:rerun-if-changed=include/GMRFLib/GMRFLibP.h");
    println!("cargo:rerun-if-changed=include/GMRFLib/pre-opt.h");
    println!("cargo:rerun-if-changed=include/GMRFLib/sha.h");
    println!("cargo:rerun-if-changed=src/gmrflib_bindings_pregen.rs");
    println!("cargo:rerun-if-changed=../gmrflib/version.h");
    println!("cargo:rerun-if-changed=../gmrflib/hash.h");
    println!("cargo:rerun-if-changed=../gmrflib/hashP.h");
    println!("cargo:rerun-if-changed=../gmrflib/GMRFLibP.h");
    println!("cargo:rerun-if-changed=../gmrflib/graph.h");
    println!("cargo:rerun-if-changed=../gmrflib/taucs.h");
    println!("cargo:rerun-if-changed=../gmrflib/smtp-pardiso.h");
    println!("cargo:rerun-if-changed=../gmrflib/sparse-interface.h");
    println!("cargo:rerun-if-changed=../gmrflib/blockupdate.h");
    println!("cargo:rerun-if-changed=../gmrflib/approx-inference.h");
    println!("cargo:rerun-if-changed=../gmrflib/optimize.h");
    println!("cargo:rerun-if-changed=../gmrflib/pre-opt.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_file = out_dir.join("gmrflib_bindings.rs");
    let do_bindgen = env::var_os("CARGO_FEATURE_GENERATE_BINDINGS").is_some();

    if do_bindgen {
        let bindings = bindgen::Builder::default()
            .header("include/gmrflib_wrapper.h")
            .clang_arg("-Iinclude")
            .clang_arg("-I../gmrflib")
            .allowlist_function("GMRFLib_.*")
            .allowlist_type("GMRFLib_.*")
            .allowlist_type("_IO_FILE")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate gmrflib bindings with bindgen");

        bindings
            .write_to_file(&out_file)
            .expect("Unable to write bindgen output");
    } else {
        fs::copy("src/gmrflib_bindings_pregen.rs", &out_file)
            .expect("Unable to copy pre-generated gmrflib bindings");
    }
}

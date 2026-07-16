// Bindgen preserves C names (GMRFLib_*_tp, taucs_*, __off_t, …).
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_imports,
    unnecessary_transmutes,
    clippy::all
)]
mod raw {
    include!(concat!(env!("OUT_DIR"), "/gmrflib_bindings.rs"));
}

use std::ptr::NonNull;

pub use raw::{GMRFLib_ai_store_tp, GMRFLib_blockupdate_param_tp, GMRFLib_optimize_param_tp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GmrfLibError {
    pub code: i32,
}

impl std::fmt::Display for GmrfLibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GMRFLib returned non-zero status {}", self.code)
    }
}

impl std::error::Error for GmrfLibError {}

fn status_to_result(status: i32) -> Result<(), GmrfLibError> {
    if status == 0 {
        Ok(())
    } else {
        Err(GmrfLibError { code: status })
    }
}

pub fn smtp_name(smtp: i32) -> &'static str {
    match smtp {
        1 => "band",
        2 => "taucs",
        3 => "pardiso",
        4 => "default",
        _ => "THIS SHOULD NOT HAPPEN",
    }
}

pub fn version_stdout() -> Result<(), GmrfLibError> {
    let status = unsafe { raw::GMRFLib_version(std::ptr::null_mut()) };
    status_to_result(status)
}

pub fn default_optimize_param() -> Result<NonNull<GMRFLib_optimize_param_tp>, GmrfLibError> {
    let mut ptr: *mut GMRFLib_optimize_param_tp = std::ptr::null_mut();
    let status = unsafe { raw::GMRFLib_default_optimize_param(&mut ptr) };
    status_to_result(status)?;
    NonNull::new(ptr).ok_or(GmrfLibError { code: -1 })
}

pub fn default_blockupdate_param() -> Result<NonNull<GMRFLib_blockupdate_param_tp>, GmrfLibError> {
    let mut ptr: *mut GMRFLib_blockupdate_param_tp = std::ptr::null_mut();
    let status = unsafe { raw::GMRFLib_default_blockupdate_param(&mut ptr) };
    status_to_result(status)?;
    NonNull::new(ptr).ok_or(GmrfLibError { code: -1 })
}

pub struct AiStore {
    ptr: NonNull<GMRFLib_ai_store_tp>,
}

impl AiStore {
    /// # Safety
    ///
    /// `ptr` must be null or a valid pointer uniquely owned by the caller and
    /// allocated by GMRFLib (freed via `GMRFLib_free_ai_store` on drop).
    pub unsafe fn from_raw_owned(ptr: *mut GMRFLib_ai_store_tp) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    pub fn as_ptr(&self) -> *mut GMRFLib_ai_store_tp {
        self.ptr.as_ptr()
    }
}

impl Drop for AiStore {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::GMRFLib_free_ai_store(self.ptr.as_ptr());
        }
    }
}

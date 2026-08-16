//! Shared posterior helpers used after a fit (copy constraint, lincomb, sampling).

/// Tightness of the soft copy constraint \(x_{\mathrm{copy}} \approx \beta x_{\mathrm{src}}\).
pub const COPY_PRECISION: f64 = 1e6;

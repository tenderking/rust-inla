//! Named option bag resolved in Rust, so a new control is one field here plus
//! (optionally) one alias — not a new positional argument threaded through both
//! FFI signatures.
//!
//! Front-ends collect user options into `(key, value)` pairs and call
//! [`resolve_compute_options`]. Unknown keys are rejected rather than silently
//! ignored, which is what previously let `control.compute` exist in Python but
//! not in R.

/// Loosely typed option value coming from R (`list`) or Python (`dict`).
#[derive(Debug, Clone, PartialEq)]
pub enum OptionValue {
    Bool(bool),
    Num(f64),
    Text(String),
    Nums(Vec<f64>),
}

impl OptionValue {
    fn as_bool(&self, key: &str) -> Result<bool, String> {
        match self {
            OptionValue::Bool(b) => Ok(*b),
            OptionValue::Num(v) => Ok(*v != 0.0),
            other => Err(format!("option '{key}': expected logical, got {other:?}")),
        }
    }

    fn as_f64(&self, key: &str) -> Result<f64, String> {
        match self {
            OptionValue::Num(v) => Ok(*v),
            OptionValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            other => Err(format!("option '{key}': expected numeric, got {other:?}")),
        }
    }

    fn as_text(&self, key: &str) -> Result<String, String> {
        match self {
            OptionValue::Text(s) => Ok(s.clone()),
            other => Err(format!("option '{key}': expected string, got {other:?}")),
        }
    }

    /// `true` ⇒ all indices, `false`/absent ⇒ none, numeric vector ⇒ those indices.
    fn as_index_selection(&self, key: &str) -> Result<IndexSelection, String> {
        match self {
            OptionValue::Bool(true) => Ok(IndexSelection::All),
            OptionValue::Bool(false) => Ok(IndexSelection::None),
            OptionValue::Nums(v) => {
                let mut idx = Vec::with_capacity(v.len());
                for &x in v {
                    if x < 0.0 || x.fract() != 0.0 {
                        return Err(format!(
                            "option '{key}': indices must be non-negative integers, got {x}"
                        ));
                    }
                    idx.push(x as usize);
                }
                Ok(IndexSelection::Some(idx))
            }
            OptionValue::Num(x) => {
                if *x < 0.0 || x.fract() != 0.0 {
                    return Err(format!(
                        "option '{key}': indices must be non-negative integers, got {x}"
                    ));
                }
                Ok(IndexSelection::Some(vec![*x as usize]))
            }
            other => Err(format!(
                "option '{key}': expected logical or numeric indices, got {other:?}"
            )),
        }
    }
}

/// Which indices to return mixture marginals for.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum IndexSelection {
    #[default]
    None,
    All,
    Some(Vec<usize>),
}

impl IndexSelection {
    /// Concrete 0-based indices given the dimension, or `None` when disabled.
    pub fn resolve(&self, n: usize) -> Result<Option<Vec<usize>>, String> {
        match self {
            IndexSelection::None => Ok(None),
            IndexSelection::All => Ok(Some((0..n).collect())),
            IndexSelection::Some(idx) => {
                for &i in idx {
                    if i >= n {
                        return Err(format!("marginal index {i} out of range [0, {n})"));
                    }
                }
                Ok(Some(idx.clone()))
            }
        }
    }
}

/// Resolved engine / compute settings. Every field has a statistical or engine
/// default; front-ends never invent their own.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputeOptions {
    pub strategy: String,
    pub step_or_f0: f64,
    pub deterministic: bool,
    pub fixed_prec: f64,
    pub dic: bool,
    pub waic: bool,
    pub cpo: bool,
    pub return_marginals_latent: IndexSelection,
    pub return_marginals_predictor: IndexSelection,
}

impl Default for ComputeOptions {
    fn default() -> Self {
        Self {
            strategy: "ccd".to_string(),
            step_or_f0: 1.0,
            deterministic: false,
            fixed_prec: 1e-4,
            dic: true,
            waic: true,
            cpo: true,
            return_marginals_latent: IndexSelection::None,
            return_marginals_predictor: IndexSelection::None,
        }
    }
}

/// Canonical key for a user-supplied option name.
///
/// Accepts R-style dots and Python-style underscores, plus classic R-INLA names.
fn canonical_key(key: &str) -> Option<&'static str> {
    let k = key.trim().to_ascii_lowercase().replace('.', "_");
    Some(match k.as_str() {
        "strategy" | "int_strategy" => "strategy",
        "step_or_f0" | "step" | "f0" => "step_or_f0",
        "deterministic" => "deterministic",
        "fixed_prec" | "prec_intercept" => "fixed_prec",
        "dic" => "dic",
        "waic" => "waic",
        "cpo" => "cpo",
        "return_marginals_latent" | "return_marginals_random" | "config" => {
            "return_marginals_latent"
        }
        "return_marginals_predictor" => "return_marginals_predictor",
        _ => return None,
    })
}

/// Validate + fill defaults for a named option bag.
///
/// Unknown keys are an error: silently dropping them is how front-ends drift.
pub fn resolve_compute_options(pairs: &[(String, OptionValue)]) -> Result<ComputeOptions, String> {
    let mut out = ComputeOptions::default();
    for (raw_key, value) in pairs {
        let key = canonical_key(raw_key).ok_or_else(|| {
            format!(
                "unknown control option '{raw_key}' (known: strategy, step_or_f0, deterministic, \
                 fixed_prec, dic, waic, cpo, return_marginals_latent, return_marginals_predictor)"
            )
        })?;
        match key {
            "strategy" => {
                let s = value.as_text(key)?.trim().to_ascii_lowercase();
                if s != "ccd" && s != "grid" {
                    return Err(format!("unsupported integration strategy '{s}'"));
                }
                out.strategy = s;
            }
            "step_or_f0" => {
                let v = value.as_f64(key)?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(format!("step_or_f0 must be finite and > 0, got {v}"));
                }
                out.step_or_f0 = v;
            }
            "deterministic" => out.deterministic = value.as_bool(key)?,
            "fixed_prec" => {
                let v = value.as_f64(key)?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(format!("fixed_prec must be finite and > 0, got {v}"));
                }
                out.fixed_prec = v;
            }
            "dic" => out.dic = value.as_bool(key)?,
            "waic" => out.waic = value.as_bool(key)?,
            "cpo" => out.cpo = value.as_bool(key)?,
            "return_marginals_latent" => {
                out.return_marginals_latent = value.as_index_selection(key)?
            }
            "return_marginals_predictor" => {
                out.return_marginals_predictor = value.as_index_selection(key)?
            }
            _ => unreachable!("canonical_key returned an unhandled key"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_filled() {
        let opts = resolve_compute_options(&[]).unwrap();
        assert_eq!(opts.strategy, "ccd");
        assert!(opts.waic);
        assert_eq!(opts.return_marginals_latent, IndexSelection::None);
    }

    #[test]
    fn dotted_and_underscored_keys_agree() {
        let dotted =
            resolve_compute_options(&[("return.marginals.random".into(), OptionValue::Bool(true))])
                .unwrap();
        let under =
            resolve_compute_options(&[("return_marginals_latent".into(), OptionValue::Bool(true))])
                .unwrap();
        assert_eq!(dotted, under);
        assert_eq!(dotted.return_marginals_latent, IndexSelection::All);
    }

    #[test]
    fn unknown_key_is_rejected() {
        let err = resolve_compute_options(&[("no_such_control".into(), OptionValue::Bool(true))])
            .unwrap_err();
        assert!(err.contains("unknown control option"), "{err}");
    }

    #[test]
    fn bad_strategy_is_rejected() {
        assert!(
            resolve_compute_options(&[("strategy".into(), OptionValue::Text("mcmc".into()))])
                .is_err()
        );
    }

    #[test]
    fn index_selection_resolves() {
        let opts = resolve_compute_options(&[(
            "return_marginals_predictor".into(),
            OptionValue::Nums(vec![0.0, 2.0]),
        )])
        .unwrap();
        assert_eq!(
            opts.return_marginals_predictor.resolve(5).unwrap(),
            Some(vec![0, 2])
        );
        assert!(opts.return_marginals_predictor.resolve(2).is_err());
    }
}

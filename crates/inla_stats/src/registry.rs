//! Single source of truth for per-model metadata shared by every front-end.
//!
//! Before this module, θ-length, default θ, rank deficiency and hyperparameter
//! labels were each duplicated in `r-inla/R/inla_rs.R` and
//! `py-inla/python/inla/api.py`, so adding a latent model meant editing the same
//! table in three places (and the copies had already drifted). Bindings should
//! call [`model_metadata`] instead of keeping local tables.

use crate::plan::HyperTransformKind;
use crate::priors::{HyperPriorStack, PriorSpec};

/// Latent models understood by the structured/formula paths.
pub const SUPPORTED_MODELS: &[&str] = &[
    "iid", "rw1", "rw2", "rw2d", "ar1", "ar", "arp", "besag", "besag2", "bym", "bym2", "fgn",
    "seasonal", "crw1", "crw2", "matern2d", "spde", "fixed", "rgeneric", "copy", "iid2d", "iid3d",
    "iid4d", "iid5d",
];

/// Group (`control.group`) models understood by the Kronecker path.
pub const SUPPORTED_GROUP_MODELS: &[&str] = &["iid", "rw1", "rw2", "ar1"];

/// One internal θ coordinate: how to name it and how to map it to natural scale.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperSlotMeta {
    /// Internal-scale name, e.g. `log_precision`.
    pub internal_label: String,
    /// Natural-scale label without the effect name, e.g. `Precision`.
    pub label: String,
    pub transform: HyperTransformKind,
}

impl HyperSlotMeta {
    fn new(internal_label: &str, label: &str, transform: HyperTransformKind) -> Self {
        Self {
            internal_label: internal_label.to_string(),
            label: label.to_string(),
            transform,
        }
    }

    /// Transform tag for FFI (`exp` / `rho` / `phi` / `identity`).
    pub fn transform_tag(&self) -> &'static str {
        match self.transform {
            HyperTransformKind::Exp => "exp",
            HyperTransformKind::RhoCor1 => "rho",
            HyperTransformKind::Phi => "phi",
            HyperTransformKind::Identity => "identity",
        }
    }
}

/// Everything a front-end needs to know about a latent model instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMeta {
    pub model: String,
    /// Number of internal θ coordinates (including any group block).
    pub theta_len: usize,
    /// Starting values on the internal scale (same length as `theta_len`).
    pub default_theta: Vec<f64>,
    /// Null-space dimension for hard sum-to-zero style constraints.
    pub rank_deficiency: usize,
    /// Whether `scale.model` defaults to on (intrinsic models, per modern R-INLA).
    pub default_scale_model: bool,
    /// One entry per θ coordinate, in optimizer order.
    pub hyper: Vec<HyperSlotMeta>,
    /// Default `(prior_name, params)` per θ coordinate.
    pub default_priors: Vec<(String, Vec<f64>)>,
}

fn prec_slot() -> HyperSlotMeta {
    HyperSlotMeta::new("log_precision", "Precision", HyperTransformKind::Exp)
}

/// Metadata for `model` with an optional `order` (AR order / FGN order / rgeneric θ count)
/// and optional `group_model` from `control.group`.
///
/// `cyclic` only affects lattice models (`rw2d`), matching R-INLA's reduced null space.
pub fn model_metadata(
    model: &str,
    order: usize,
    group_model: Option<&str>,
    cyclic: bool,
) -> Result<ModelMeta, String> {
    let m = model.trim().to_ascii_lowercase();

    let mut hyper: Vec<HyperSlotMeta> = Vec::new();
    let mut default_theta: Vec<f64> = Vec::new();

    match m.as_str() {
        "fixed" => {}
        "iid" | "rw1" | "rw2" | "besag" | "besag2" | "seasonal" | "crw1" | "crw2" | "rw2d" => {
            hyper.push(prec_slot());
            default_theta.push(0.0);
        }
        "ar1" => {
            hyper.push(prec_slot());
            hyper.push(HyperSlotMeta::new(
                "logit_rho",
                "Rho",
                HyperTransformKind::RhoCor1,
            ));
            default_theta.extend_from_slice(&[0.0, 0.0]);
        }
        "ar" | "arp" => {
            let p = if order > 0 { order } else { 2 };
            hyper.push(prec_slot());
            default_theta.push(0.0);
            for i in 1..=p {
                hyper.push(HyperSlotMeta::new(
                    &format!("pacf{i}_intern"),
                    &format!("PACF{i}"),
                    HyperTransformKind::Identity,
                ));
                default_theta.push(0.0);
            }
        }
        "bym" => {
            hyper.push(HyperSlotMeta::new(
                "log_precision_spatial",
                "Precision (spatial)",
                HyperTransformKind::Exp,
            ));
            hyper.push(HyperSlotMeta::new(
                "log_precision_iid",
                "Precision (iid)",
                HyperTransformKind::Exp,
            ));
            default_theta.extend_from_slice(&[1.0, 1.0]);
        }
        "bym2" => {
            hyper.push(prec_slot());
            hyper.push(HyperSlotMeta::new(
                "logit_phi",
                "Phi",
                HyperTransformKind::Phi,
            ));
            default_theta.extend_from_slice(&[1.0, 0.0]);
        }
        "matern2d" => {
            hyper.push(prec_slot());
            hyper.push(HyperSlotMeta::new(
                "log_range",
                "Range",
                HyperTransformKind::Exp,
            ));
            default_theta.extend_from_slice(&[0.0, 0.0]);
        }
        "spde" => {
            hyper.push(HyperSlotMeta::new(
                "log_tau",
                "Tau",
                HyperTransformKind::Exp,
            ));
            hyper.push(HyperSlotMeta::new(
                "log_kappa",
                "Kappa",
                HyperTransformKind::Exp,
            ));
            default_theta.extend_from_slice(&[0.0, 0.0]);
        }
        "fgn" => {
            hyper.push(prec_slot());
            // No closed-form natural map: report the internal Hurst coordinate as such.
            hyper.push(HyperSlotMeta::new(
                "hurst_intern",
                "Hurst.intern",
                HyperTransformKind::Identity,
            ));
            // AR-mixture approximation starts away from the boundary.
            if order == 3 || order == 4 {
                default_theta.extend_from_slice(&[1.0, 2.0]);
            } else {
                default_theta.extend_from_slice(&[0.0, 0.0]);
            }
        }
        "copy" => {
            hyper.push(HyperSlotMeta::new(
                "beta",
                "Beta",
                HyperTransformKind::Identity,
            ));
            default_theta.push(1.0);
        }
        "iid2d" | "iid3d" | "iid4d" | "iid5d" => {
            let d = crate::iidkd::iidkd_dim(&m).expect("iidkd dim");
            for i in 1..=d {
                hyper.push(HyperSlotMeta::new(
                    &format!("log_precision{i}"),
                    &format!("Precision (component {i})"),
                    HyperTransformKind::Exp,
                ));
                default_theta.push(4.0);
            }
            for i in 1..=d {
                for j in (i + 1)..=d {
                    hyper.push(HyperSlotMeta::new(
                        &format!("logit_rho{i}{j}"),
                        &format!("Rho{i}:{j}"),
                        HyperTransformKind::RhoCor1,
                    ));
                    default_theta.push(if d == 2 { 4.0 } else { 0.0 });
                }
            }
        }
        "rgeneric" => {
            let n_th = if order > 0 { order } else { 1 };
            for i in 1..=n_th {
                hyper.push(HyperSlotMeta::new(
                    &format!("theta{i}"),
                    &format!("Theta{i}"),
                    HyperTransformKind::Identity,
                ));
                default_theta.push(0.0);
            }
        }
        other => return Err(format!("unknown latent model '{other}'")),
    }

    let mut default_priors: Vec<(String, Vec<f64>)> = default_prior_pairs(&m);

    if let Some(g) = group_model.filter(|g| !g.trim().is_empty()) {
        let gm = g.trim().to_ascii_lowercase();
        match gm.as_str() {
            "iid" | "rw1" | "rw2" => {
                hyper.push(HyperSlotMeta::new(
                    "log_precision_group",
                    "Group precision",
                    HyperTransformKind::Exp,
                ));
                default_theta.push(0.0);
            }
            "ar1" => {
                hyper.push(HyperSlotMeta::new(
                    "log_precision_group",
                    "Group precision",
                    HyperTransformKind::Exp,
                ));
                hyper.push(HyperSlotMeta::new(
                    "logit_rho_group",
                    "Group rho",
                    HyperTransformKind::RhoCor1,
                ));
                default_theta.extend_from_slice(&[0.0, 0.0]);
            }
            other => return Err(format!("unsupported control.group model '{other}'")),
        }
        default_priors.extend(default_prior_pairs(&gm));
    }

    // `order` carries the season length for seasonal models.
    let rank_deficiency = if m == "seasonal" {
        let season = if order >= 2 { order } else { 4 };
        season - 1
    } else {
        rank_deficiency(&m, cyclic)
    };

    Ok(ModelMeta {
        default_scale_model: default_scale_model(&m),
        model: m,
        theta_len: hyper.len(),
        default_theta,
        rank_deficiency,
        hyper,
        default_priors,
    })
}

/// Modern R-INLA scales intrinsic models so the generalized marginal variance is 1.
pub fn default_scale_model(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().as_str(),
        "rw1" | "rw2" | "rw2d" | "besag" | "besag2" | "bym" | "bym2"
    )
}

/// Null-space dimension used for hard constraints (`cyclic` shrinks lattice models).
pub fn rank_deficiency(model: &str, cyclic: bool) -> usize {
    let m = model.trim().to_ascii_lowercase();
    if m == "rw2d" {
        return if cyclic { 1 } else { 3 };
    }
    inla_math::model_rank_deficiency(&m)
}

fn default_prior_pairs(model: &str) -> Vec<(String, Vec<f64>)> {
    match HyperPriorStack::default_for_effect(model) {
        Ok(stack) => stack.priors.iter().map(prior_to_pair).collect(),
        Err(_) => Vec::new(),
    }
}

fn prior_to_pair(p: &PriorSpec) -> (String, Vec<f64>) {
    p.to_pair()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_metadata() {
        let meta = model_metadata("ar1", 0, None, false).unwrap();
        assert_eq!(meta.theta_len, 2);
        assert_eq!(meta.default_theta.len(), 2);
        assert_eq!(meta.hyper[0].label, "Precision");
        assert_eq!(meta.hyper[0].transform_tag(), "exp");
        assert_eq!(meta.hyper[1].transform_tag(), "rho");
        assert_eq!(meta.rank_deficiency, 0);
    }

    #[test]
    fn theta_len_matches_default_theta_for_all_models() {
        for m in SUPPORTED_MODELS {
            let order = if *m == "ar" || *m == "arp" { 2 } else { 0 };
            let meta = model_metadata(m, order, None, false).unwrap();
            assert_eq!(
                meta.theta_len,
                meta.default_theta.len(),
                "model {m}: theta_len {} != default_theta {}",
                meta.theta_len,
                meta.default_theta.len()
            );
            assert_eq!(meta.theta_len, meta.hyper.len(), "model {m}: hyper len");
        }
    }

    #[test]
    fn group_model_appends_slots() {
        let plain = model_metadata("besag", 0, None, false).unwrap();
        let grouped = model_metadata("besag", 0, Some("ar1"), false).unwrap();
        assert_eq!(grouped.theta_len, plain.theta_len + 2);
        assert_eq!(grouped.hyper.last().unwrap().transform_tag(), "rho");
    }

    #[test]
    fn arp_order_controls_theta_len() {
        let meta = model_metadata("arp", 3, None, false).unwrap();
        assert_eq!(meta.theta_len, 4);
        assert_eq!(meta.hyper[3].label, "PACF3");
    }

    #[test]
    fn rw2d_cyclic_reduces_rank() {
        assert_eq!(
            model_metadata("rw2d", 0, None, false)
                .unwrap()
                .rank_deficiency,
            3
        );
        assert_eq!(
            model_metadata("rw2d", 0, None, true)
                .unwrap()
                .rank_deficiency,
            1
        );
    }

    #[test]
    fn seasonal_rank_follows_season_length() {
        assert_eq!(
            model_metadata("seasonal", 12, None, false)
                .unwrap()
                .rank_deficiency,
            11
        );
        // Unspecified season falls back to the quarterly default.
        assert_eq!(
            model_metadata("seasonal", 0, None, false)
                .unwrap()
                .rank_deficiency,
            3
        );
    }

    #[test]
    fn intrinsic_models_scale_by_default() {
        assert!(
            model_metadata("rw2", 0, None, false)
                .unwrap()
                .default_scale_model
        );
        assert!(
            model_metadata("besag", 0, None, false)
                .unwrap()
                .default_scale_model
        );
        assert!(
            !model_metadata("ar1", 0, None, false)
                .unwrap()
                .default_scale_model
        );
        assert!(
            !model_metadata("iid", 0, None, false)
                .unwrap()
                .default_scale_model
        );
    }

    #[test]
    fn unknown_model_is_rejected() {
        assert!(model_metadata("does_not_exist", 0, None, false).is_err());
    }

    #[test]
    fn copy_metadata() {
        let meta = model_metadata("copy", 0, None, false).unwrap();
        assert_eq!(meta.theta_len, 1);
        assert_eq!(meta.default_theta, vec![1.0]);
        assert_eq!(meta.hyper[0].label, "Beta");
        assert_eq!(meta.hyper[0].transform_tag(), "identity");
        assert_eq!(meta.rank_deficiency, 0);
        assert!(!meta.default_scale_model);
    }

    #[test]
    fn iid2d_metadata() {
        let meta = model_metadata("iid2d", 0, None, false).unwrap();
        assert_eq!(meta.theta_len, 3);
        assert_eq!(meta.default_theta.len(), 3);
        assert_eq!(meta.hyper.len(), 3);
        assert_eq!(meta.rank_deficiency, 0);
        assert_eq!(meta.hyper[0].transform_tag(), "exp");
        assert_eq!(meta.hyper[2].transform_tag(), "rho");
        assert_eq!(meta.hyper[2].label, "Rho1:2");
        assert_eq!(meta.default_priors[0].0, "wishart2d");
    }
}

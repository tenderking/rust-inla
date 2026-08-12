//! Shared multi-effect θ → Q / constraints / default priors.
//!
//! Both `r-inla` and `py-inla` should assemble host-side metadata into
//! [`StructuredEffect`] slices and call these helpers instead of duplicating
//! model `match` arms in each binding.

use inla_math::{
    ConstraintSpec, CscMatrix, block_diag_csc, identity_csc, model_rank_deficiency,
    plane_constraint_2d, scale_csc, scale_model_csc, seasonal_constraint, sum_to_zero_constraint,
};

use crate::ar1::ar1_precision_csc;
use crate::arp::arp_precision_csc;
use crate::besag::{besag_precision_csc, bym_precision_csc, bym2_precision_csc};
use crate::crw::{crw1_precision_csc, crw2_precision_csc};
use crate::fgn::{fgn_approx_precision_csc, fgn_hurst_from_intern};
use crate::latent_models::{
    fgn_precision_csc, iid_precision_csc, rw1_precision_csc, rw2_precision_csc,
    seasonal_precision_csc,
};
use crate::matern2d::matern2d_precision_csc;
use crate::priors::{HyperPriorStack, PriorSpec};
use crate::rw2d::rw2d_precision_csc;

/// One latent block in a structured (multi-effect) model.
///
/// Host languages fill this from formula/`f()` metadata. No R/Python types.
#[derive(Debug, Clone)]
pub struct StructuredEffect {
    pub model: String,
    pub n: usize,
    pub scale_model: bool,
    pub theta_len: usize,
    /// Season length, FGN order, or ±nrow for `rw2d`/`matern2d` (negative ⇒ cyclic).
    pub order: i32,
    pub adj: Option<Vec<Vec<usize>>>,
    pub positions: Option<Vec<f64>>,
    pub crw2_layout: String,
    pub nrow: usize,
    pub ncol: usize,
    pub cyclic: bool,
    pub matern_nu: usize,
}

impl StructuredEffect {
    pub fn simple(model: impl Into<String>, n: usize, theta_len: usize) -> Self {
        Self {
            model: model.into(),
            n,
            scale_model: false,
            theta_len,
            order: 0,
            adj: None,
            positions: None,
            crw2_layout: "simple".into(),
            nrow: 0,
            ncol: 0,
            cyclic: false,
            matern_nu: 1,
        }
    }

    pub fn model_key(&self) -> String {
        self.model.to_ascii_lowercase()
    }
}

fn maybe_scale(q: CscMatrix, scale_model: bool) -> Result<CscMatrix, String> {
    if scale_model {
        scale_model_csc(&q)
    } else {
        Ok(q)
    }
}

fn apply_tau(q: &CscMatrix, tau: f64) -> Result<CscMatrix, String> {
    scale_csc(q, tau)
}

fn one_block(effect: &StructuredEffect, th: &[f64], fixed_prec: f64) -> Result<CscMatrix, String> {
    let typ = effect.model_key();
    let n_e = effect.n;
    match typ.as_str() {
        "fixed" => identity_csc(n_e, fixed_prec),
        "iid" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let q0 = maybe_scale(iid_precision_csc(n_e, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "rw1" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let q0 = maybe_scale(rw1_precision_csc(n_e, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "rw2" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let q0 = maybe_scale(rw2_precision_csc(n_e, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "rw2d" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let (nrow, ncol, cyclic) = rw2d_dims(effect)?;
            let q0 = maybe_scale(
                rw2d_precision_csc(nrow, ncol, 1.0, cyclic, false)?,
                effect.scale_model,
            )?;
            apply_tau(&q0, tau)
        }
        "matern2d" => {
            if th.len() < 2 {
                return Err("matern2d needs theta=[log_prec, log_range]".into());
            }
            let prec = th[0].exp();
            let range = th[1].exp();
            let (nrow, ncol, cyclic) = rw2d_dims(effect)?;
            let nu = if effect.matern_nu > 0 {
                effect.matern_nu
            } else {
                1
            };
            matern2d_precision_csc(nrow, ncol, nu, range, prec, cyclic)
        }
        "bym" => {
            if th.len() < 2 {
                return Err("bym needs theta=[log_tau_spatial, log_tau_iid]".into());
            }
            let adj = effect
                .adj
                .as_ref()
                .ok_or_else(|| "bym missing adj".to_string())?;
            if adj.len() * 2 != n_e {
                return Err(format!(
                    "bym adj length {} ⇒ latent {}, got n={n_e}",
                    adj.len(),
                    adj.len() * 2
                ));
            }
            bym_precision_csc(adj, th[0].exp(), th[1].exp())
        }
        "bym2" => {
            if th.len() < 2 {
                return Err("bym2 needs theta=[log_tau, logit_phi]".into());
            }
            let adj = effect
                .adj
                .as_ref()
                .ok_or_else(|| "bym2 missing adj".to_string())?;
            if adj.len() != n_e {
                return Err(format!("bym2 adj length {} != effect n {n_e}", adj.len()));
            }
            let tau = th[0].exp();
            let phi = (1.0 / (1.0 + (-th[1]).exp())).clamp(1e-6, 1.0 - 1e-6);
            bym2_precision_csc(adj, tau, phi)
        }
        "seasonal" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let season = if effect.order > 0 {
                effect.order as usize
            } else {
                4
            };
            let q0 = maybe_scale(
                seasonal_precision_csc(n_e, season, 1.0, true)?,
                effect.scale_model,
            )?;
            apply_tau(&q0, tau)
        }
        "ar1" => {
            if th.len() < 2 {
                return Err("ar1 needs theta=[log_tau, logit_rho]".into());
            }
            let tau = th[0].exp();
            let rho = (2.0 / (1.0 + (-th[1]).exp()) - 1.0).clamp(-0.999, 0.999);
            let q0 = maybe_scale(ar1_precision_csc(n_e, rho, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "ar" | "arp" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let pacf: Vec<f64> = if th.len() > 1 {
                th[1..].iter().map(|&v| (v * 0.5).tanh()).collect()
            } else {
                vec![0.0]
            };
            let q0 = maybe_scale(arp_precision_csc(n_e, &pacf, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "crw1" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let positions = effect
                .positions
                .clone()
                .unwrap_or_else(|| (0..n_e).map(|i| i as f64).collect());
            let q0 = maybe_scale(crw1_precision_csc(&positions, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "crw2" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let positions = effect
                .positions
                .clone()
                .unwrap_or_else(|| (0..n_e).map(|i| i as f64).collect());
            let layout = if effect.crw2_layout.is_empty() {
                "simple"
            } else {
                effect.crw2_layout.as_str()
            };
            let q0 = maybe_scale(
                crw2_precision_csc(&positions, 1.0, layout)?,
                effect.scale_model,
            )?;
            apply_tau(&q0, tau)
        }
        "besag" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let adj = effect
                .adj
                .as_ref()
                .ok_or_else(|| "besag missing adj".to_string())?;
            if adj.len() != n_e {
                return Err(format!(
                    "besag adj length {} != effect n {n_e}",
                    adj.len()
                ));
            }
            let q0 = maybe_scale(besag_precision_csc(adj, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "fgn" => {
            let order = if effect.order > 0 {
                effect.order as usize
            } else {
                0
            };
            if order == 3 || order == 4 {
                if th.len() < 2 {
                    return Err("fgn approx needs [log_tau, H_intern]".into());
                }
                let tau = th[0].exp();
                let hurst = fgn_hurst_from_intern(th[1]);
                let n_time = n_e / (order + 1);
                let q0 = maybe_scale(
                    fgn_approx_precision_csc(n_time, hurst, 1.0, order, 1e8)?,
                    effect.scale_model,
                )?;
                apply_tau(&q0, tau)
            } else {
                if th.len() < 2 {
                    return Err("fgn needs [log_tau, logit_H]".into());
                }
                let tau = th[0].exp();
                let hurst = 1.0 / (1.0 + (-th[1]).exp());
                let q0 = maybe_scale(fgn_precision_csc(n_e, hurst, 1.0)?, effect.scale_model)?;
                apply_tau(&q0, tau)
            }
        }
        other => Err(format!("unsupported effect type: {other}")),
    }
}

fn rw2d_dims(effect: &StructuredEffect) -> Result<(usize, usize, bool), String> {
    if effect.nrow > 0 && effect.ncol > 0 {
        return Ok((effect.nrow, effect.ncol, effect.cyclic));
    }
    // R encoding: order = ±nrow (negative ⇒ cyclic); ncol = n / nrow.
    let raw = effect.order;
    let cyclic = raw < 0 || effect.cyclic;
    let nrow = raw.unsigned_abs() as usize;
    if nrow == 0 || !effect.n.is_multiple_of(nrow) {
        return Err(format!(
            "rw2d/matern2d: order (±nrow)={raw} incompatible with n={}",
            effect.n
        ));
    }
    Ok((nrow, effect.n / nrow, cyclic))
}

/// Block-diagonal prior precision for concatenated θ across [`StructuredEffect`]s.
pub fn build_structured_precision(
    effects: &[StructuredEffect],
    theta: &[f64],
    fixed_prec: f64,
) -> Result<CscMatrix, String> {
    let expected: usize = effects.iter().map(|e| e.theta_len).sum();
    if theta.len() != expected {
        return Err(format!(
            "theta length {} != sum(effect theta_len) {expected}",
            theta.len()
        ));
    }
    let mut blocks = Vec::with_capacity(effects.len());
    let mut off = 0usize;
    for effect in effects {
        let tlen = effect.theta_len;
        let th = if tlen == 0 {
            &[][..]
        } else {
            &theta[off..off + tlen]
        };
        off += tlen;
        blocks.push(one_block(effect, th, fixed_prec)?);
    }
    block_diag_csc(&blocks)
}

/// Default hyperprior stack (skips `fixed` blocks).
pub fn structured_prior_stack(effects: &[StructuredEffect]) -> HyperPriorStack {
    let mut priors = Vec::new();
    for effect in effects {
        let m = effect.model_key();
        if m == "fixed" {
            continue;
        }
        match HyperPriorStack::default_for_effect(&m) {
            Ok(s) => priors.extend(s.priors),
            Err(_) => priors.push(PriorSpec::gaussian(0.0, 0.1)),
        }
    }
    HyperPriorStack::new(priors)
}

/// Hard linear constraints for intrinsic / BYM / rw2d blocks.
pub fn structured_constraints(effects: &[StructuredEffect]) -> Result<Option<ConstraintSpec>, String> {
    let full_n: usize = effects.iter().map(|e| e.n).sum();
    let mut stacked: Option<ConstraintSpec> = None;
    let mut offset = 0usize;
    for effect in effects {
        let typ = effect.model_key();
        let n_e = effect.n;
        if typ == "rw2d" {
            let (nrow, ncol, cyclic) = rw2d_dims(effect)?;
            let block = if cyclic {
                sum_to_zero_constraint(n_e, 1)?
            } else {
                plane_constraint_2d(nrow, ncol)?
            };
            let embedded = block.embed(full_n, offset)?;
            stacked = Some(match stacked {
                None => embedded,
                Some(prev) => prev.vstack(&embedded)?,
            });
            offset += n_e;
            continue;
        }
        if typ == "seasonal" {
            let season = usize::try_from(effect.order.max(2)).unwrap_or(4);
            let block = seasonal_constraint(n_e, season.min(n_e))?;
            let embedded = block.embed(full_n, offset)?;
            stacked = Some(match stacked {
                None => embedded,
                Some(prev) => prev.vstack(&embedded)?,
            });
            offset += n_e;
            continue;
        }
        if typ == "bym" {
            let n_sp = effect
                .adj
                .as_ref()
                .map(|a| a.len())
                .unwrap_or(n_e / 2);
            let block = sum_to_zero_constraint(n_sp, 1)?;
            let embedded = block.embed(full_n, offset)?;
            stacked = Some(match stacked {
                None => embedded,
                Some(prev) => prev.vstack(&embedded)?,
            });
            offset += n_e;
            continue;
        }
        let k = model_rank_deficiency(&typ);
        if k > 0 {
            let block = sum_to_zero_constraint(n_e, k)?;
            let embedded = block.embed(full_n, offset)?;
            stacked = Some(match stacked {
                None => embedded,
                Some(prev) => prev.vstack(&embedded)?,
            });
        }
        offset += n_e;
    }
    Ok(stacked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_block_precision_finite() {
        let effects = [StructuredEffect::simple("ar1", 8, 2)];
        let q = build_structured_precision(&effects, &[0.0, 0.0], 1e-4).unwrap();
        assert_eq!(q.rows(), 8);
        assert!(q.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn iid_rw2_block_diag() {
        let effects = [
            StructuredEffect::simple("iid", 4, 1),
            StructuredEffect::simple("rw2", 5, 1),
        ];
        let q = build_structured_precision(&effects, &[0.0, 0.0], 1e-4).unwrap();
        assert_eq!(q.rows(), 9);
    }

    #[test]
    fn seasonal_constraints_kill_the_whole_null_space() {
        let season = 4usize;
        let n = 24usize;
        let mut effect = StructuredEffect::simple("seasonal", n, 1);
        effect.order = season as i32;

        let c = structured_constraints(&[effect.clone()]).unwrap().unwrap();
        assert_eq!(c.k, season - 1);

        // Every constraint row must annihilate Q: A lives in the null space.
        let q = build_structured_precision(&[effect], &[0.0], 1e-4).unwrap();
        let dense = q.to_dense();
        for r in 0..c.k {
            let row = &c.a[r * n..(r + 1) * n];
            for i in 0..n {
                let qx: f64 = (0..n).map(|j| dense[[i, j]] * row[j]).sum();
                assert!(qx.abs() < 1e-8, "row {r} entry {i} = {qx}");
            }
        }
    }

    #[test]
    fn rw2_constraints_rank2() {
        let effects = [StructuredEffect::simple("rw2", 6, 1)];
        let c = structured_constraints(&effects).unwrap().unwrap();
        assert_eq!(c.k, 2);
        assert_eq!(c.n, 6);
    }
}

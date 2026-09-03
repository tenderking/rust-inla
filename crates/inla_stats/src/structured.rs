//! Shared multi-effect θ → Q / constraints / default priors.
//!
//! Both `r-inla` and `py-inla` should assemble host-side metadata into
//! [`StructuredEffect`] slices and call these helpers instead of duplicating
//! model `match` arms in each binding.

use inla_math::{
    ConstraintMethod, ConstraintSpec, CscMatrix, add_csc, block_diag_csc, identity_csc,
    kronecker_csc, model_rank_deficiency, plane_constraint_2d, scale_csc, scale_model_csc,
    seasonal_constraint, sparse_from_triplets, sum_to_zero_constraint,
};

use crate::posterior::COPY_PRECISION;

use crate::ar1::ar1_precision_csc;
use crate::arp::arp_precision_csc;
use crate::besag::{besag_precision_csc, bym_precision_csc, bym2_precision_csc, graph_components};
use crate::crw::{crw1_precision_csc, crw2_precision_csc};
use crate::fgn::{fgn_approx_precision_csc, fgn_hurst_from_intern};
use crate::iidkd::{iidkd_dim, iidkd_precision_csc};
use crate::latent_models::{
    fgn_precision_csc, iid_precision_csc, rw1_precision_csc, rw2_precision_csc,
    seasonal_precision_csc,
};
use crate::matern2d::matern2d_precision_csc;
use crate::priors::HyperPriorStack;
use crate::registry::{SUPPORTED_MODELS, model_metadata};
use crate::rw2d::rw2d_precision_csc;
use crate::spde::{spde_params_from_theta, spde_precision_csc};
use inla_fmesher::{Triangle, Vertex2, build_mesh1d, build_mesh2d};

/// One latent block in a structured (multi-effect) model.
///
/// Host languages fill this from formula/`f()` metadata. No R/Python types.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredEffect {
    pub model: String,
    pub n: usize,
    pub scale_model: bool,
    pub theta_len: usize,
    /// AR(p) order or FGN mixture order. Not used for lattice size or season length.
    pub order: i32,
    /// Seasonal period; `0` means the model default (4).
    pub season: usize,
    pub adj: Option<Vec<Vec<usize>>>,
    pub positions: Option<Vec<f64>>,
    pub crw2_layout: String,
    pub nrow: usize,
    pub ncol: usize,
    pub cyclic: bool,
    pub matern_nu: usize,
    /// Main-model latent dimension when `group_model` is present.
    pub n_main: usize,
    /// Optional `control.group` model; θ and latent ordering are main then group.
    pub group_model: Option<String>,
    pub group_n: usize,
    pub group_scale_model: bool,
    /// Index of the source effect when `model == "copy"` (source must appear first).
    pub copy_of: Option<usize>,
    /// SPDE mesh when `model == "spde"` (boxed so [`StructuredEffect`] stays small).
    pub mesh: Option<Box<SpdeMesh>>,
}

/// Mesh used to build SPDE `Q(θ)` (2D triangles, or 1D knots via [`Self::loc_1d`]).
///
/// Barrier / anisotropy are **fixed geometry**, not extra θ. `θ` stays
/// `[log τ, log κ]`. Empty `barrier_triangles` and `diffusion = [1, 0, 1]`
/// recover the isotropic Matérn SPDE.
#[derive(Debug, Clone, PartialEq)]
pub struct SpdeMesh {
    pub vertices: Vec<(f64, f64)>,
    pub triangles: Vec<[usize; 3]>,
    /// Ordered 1D knot locations. When set, FEM uses the 1D assembler.
    pub loc_1d: Option<Vec<f64>>,
    /// 0-based triangle indices treated as a spatial barrier (2D only).
    pub barrier_triangles: Vec<usize>,
    /// Range multiplier on barrier triangles (classic Bakka default ~0.1).
    pub range_fraction: f64,
    /// Anisotropic diffusion `[hxx, hxy, hyy]`; identity is `[1, 0, 1]`.
    pub diffusion: [f64; 3],
}

impl SpdeMesh {
    pub fn isotropic_2d(vertices: Vec<(f64, f64)>, triangles: Vec<[usize; 3]>) -> Self {
        Self {
            vertices,
            triangles,
            loc_1d: None,
            barrier_triangles: Vec::new(),
            range_fraction: 1.0,
            diffusion: [1.0, 0.0, 1.0],
        }
    }

    pub fn knots_1d(loc: Vec<f64>) -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            loc_1d: Some(loc),
            barrier_triangles: Vec::new(),
            range_fraction: 1.0,
            diffusion: [1.0, 0.0, 1.0],
        }
    }
}

impl StructuredEffect {
    pub fn simple(model: impl Into<String>, n: usize, theta_len: usize) -> Self {
        Self {
            model: model.into(),
            n,
            scale_model: false,
            theta_len,
            order: 0,
            season: 0,
            adj: None,
            positions: None,
            crw2_layout: "simple".into(),
            nrow: 0,
            ncol: 0,
            cyclic: false,
            matern_nu: 1,
            n_main: 0,
            group_model: None,
            group_n: 0,
            group_scale_model: false,
            copy_of: None,
            mesh: None,
        }
    }

    pub fn model_key(&self) -> String {
        self.model.to_ascii_lowercase()
    }

    fn season_len(&self) -> usize {
        if self.season > 0 {
            self.season
        } else if self.order > 0 {
            self.order as usize
        } else {
            4
        }
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

fn spde_fem(effect: &StructuredEffect) -> Result<(inla_fmesher::FemBlocks, usize), String> {
    let mesh = effect
        .mesh
        .as_deref()
        .ok_or_else(|| "spde missing mesh vertices".to_string())?;
    if let Some(loc) = mesh.loc_1d.as_ref() {
        if loc.is_empty() {
            return Err("spde 1D mesh has no knots".into());
        }
        if !mesh.barrier_triangles.is_empty() {
            return Err("barrier SPDE is only supported on 2D triangular meshes".into());
        }
        if mesh.diffusion != [1.0, 0.0, 1.0] {
            return Err("anisotropic SPDE is only supported on 2D triangular meshes".into());
        }
        let m1 = build_mesh1d(loc.clone())?;
        let n = m1.n();
        Ok((m1.assemble_fem_blocks(), n))
    } else {
        if mesh.vertices.is_empty() {
            return Err("spde mesh has no vertices".into());
        }
        if mesh.triangles.is_empty() {
            return Err("spde mesh has no triangles".into());
        }
        let vertices = mesh
            .vertices
            .iter()
            .map(|&(x, y)| Vertex2 { x, y })
            .collect::<Vec<_>>();
        let triangles = mesh
            .triangles
            .iter()
            .copied()
            .map(Triangle)
            .collect::<Vec<_>>();
        let m2 = build_mesh2d(vertices, triangles)?;
        let n = m2.vertices.len();
        let fem =
            m2.assemble_fem_geometry(&mesh.barrier_triangles, mesh.range_fraction, mesh.diffusion)?;
        Ok((fem, n))
    }
}

/// R-INLA `f(..., diagonal=)` default when `constr=TRUE`.
///
/// RW2 is rank-2, but classic INLA only hard-constrains the constant
/// (`constr=TRUE`). The linear-trend null space is regularized by this
/// small ridge, not by an `extraconstr`. A hard linear constraint sends
/// spatially structured covariates into Besag.
const RW2_CONSTR_DIAGONAL: f64 = 1e-4;

/// Discrete second differences or Lindgren & Rue (2008) irregular Galerkin precision.
///
/// When `positions` are provided, computes the irregular RW2 precision matrix
/// (`crw2_precision_csc(..., "simple")`). If unit spaced, this reduces identically
/// to discrete `D₂'D₂`.
fn rw2_structure_csc(n: usize, positions: Option<&[f64]>) -> Result<CscMatrix, String> {
    match positions {
        Some(pos) if pos.len() == n => crw2_precision_csc(pos, 1.0, "simple"),
        Some(pos) => Err(format!(
            "rw2 positions length {} != effect n {n}",
            pos.len()
        )),
        None => rw2_precision_csc(n, 1.0),
    }
}

fn rw1_structure_csc(n: usize, positions: Option<&[f64]>) -> Result<CscMatrix, String> {
    match positions {
        Some(pos) if pos.len() == n => crw1_precision_csc(pos, 1.0),
        Some(pos) => Err(format!(
            "rw1 positions length {} != effect n {n}",
            pos.len()
        )),
        None => rw1_precision_csc(n, 1.0),
    }
}

fn one_block(effect: &StructuredEffect, th: &[f64], fixed_prec: f64) -> Result<CscMatrix, String> {
    let typ = effect.model_key();
    let n_e = effect.n;
    let q = match typ.as_str() {
        "fixed" => identity_csc(n_e, fixed_prec),
        "copy" => identity_csc(n_e, COPY_PRECISION),
        "iid" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let q0 = maybe_scale(iid_precision_csc(n_e, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        "rw1" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let q0 = maybe_scale(
                rw1_structure_csc(n_e, effect.positions.as_deref())?,
                effect.scale_model,
            )?;
            apply_tau(&q0, tau)
        }
        "rw2" => {
            let tau = th.first().copied().unwrap_or(0.0).exp();
            let q0 = maybe_scale(
                rw2_structure_csc(n_e, effect.positions.as_deref())?,
                effect.scale_model,
            )?;
            let q_tau = apply_tau(&q0, tau)?;
            add_csc(&q_tau, &identity_csc(n_e, RW2_CONSTR_DIAGONAL)?)
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
            let season = effect.season_len();
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
            let layout = if effect.crw2_layout.is_empty() {
                "simple"
            } else {
                effect.crw2_layout.as_str()
            };
            let n_knots = if layout == "simple" {
                n_e
            } else {
                if !n_e.is_multiple_of(2) {
                    return Err(format!(
                        "crw2 layout '{layout}' requires an even latent size"
                    ));
                }
                n_e / 2
            };
            let positions = effect
                .positions
                .clone()
                .unwrap_or_else(|| (0..n_knots).map(|i| i as f64).collect());
            if positions.len() != n_knots {
                return Err(format!(
                    "crw2 layout '{layout}': positions length {} != knot count {n_knots}",
                    positions.len()
                ));
            }
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
                return Err(format!("besag adj length {} != effect n {n_e}", adj.len()));
            }
            let q0 = maybe_scale(besag_precision_csc(adj, 1.0)?, effect.scale_model)?;
            apply_tau(&q0, tau)
        }
        // Exact Q is dense Trench (small n). order=3/4 is the sparse AR-mixture.
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
                    return Err("fgn needs [log_tau, H_intern]".into());
                }
                let tau = th[0].exp();
                let hurst = fgn_hurst_from_intern(th[1]);
                let q0 = maybe_scale(fgn_precision_csc(n_e, hurst, 1.0)?, effect.scale_model)?;
                apply_tau(&q0, tau)
            }
        }
        "iid2d" | "iid3d" | "iid4d" | "iid5d" => {
            let d = iidkd_dim(&typ).ok_or_else(|| format!("iidkd: bad model {typ}"))?;
            iidkd_precision_csc(n_e, d, th)
        }
        "spde" => {
            let (tau, kappa) = spde_params_from_theta(th)?;
            let (fem, n_v) = spde_fem(effect)?;
            if n_v != n_e {
                return Err(format!("spde mesh vertices {n_v} != effect n {n_e}"));
            }
            spde_precision_csc(&fem, kappa, tau)
        }
        other => Err(format!("unsupported effect type: {other}")),
    }?;
    if q.rows() != n_e || q.cols() != n_e {
        return Err(format!(
            "effect {}: Q is {}x{}, expected {n_e}x{n_e}",
            typ,
            q.rows(),
            q.cols()
        ));
    }
    Ok(q)
}

fn effect_block(
    effect: &StructuredEffect,
    th: &[f64],
    fixed_prec: f64,
) -> Result<CscMatrix, String> {
    let Some(group_model) = effect.group_model.as_deref() else {
        return one_block(effect, th, fixed_prec);
    };
    if effect.model_key() == "copy" {
        return Err("copy effects cannot also define a group model".into());
    }
    let group_model = group_model.trim().to_ascii_lowercase();
    if !crate::registry::SUPPORTED_GROUP_MODELS.contains(&group_model.as_str()) {
        return Err(format!("unsupported control.group model '{group_model}'"));
    }
    if effect.n_main == 0 || effect.group_n == 0 || effect.n != effect.n_main * effect.group_n {
        return Err(format!(
            "grouped effect {}: n={} must equal n_main={} * group_n={}",
            effect.model_key(),
            effect.n,
            effect.n_main,
            effect.group_n
        ));
    }

    let order = usize::try_from(effect.order.max(0)).unwrap_or(0);
    let main_meta = model_metadata(&effect.model_key(), order, None, effect.cyclic)?;
    let group_meta = model_metadata(&group_model, 0, None, false)?;
    let expected = main_meta.theta_len + group_meta.theta_len;
    if th.len() != expected {
        return Err(format!(
            "grouped effect {}: theta length {} != main {} + group {}",
            effect.model_key(),
            th.len(),
            main_meta.theta_len,
            group_meta.theta_len
        ));
    }

    let mut main = effect.clone();
    main.n = effect.n_main;
    main.theta_len = main_meta.theta_len;
    main.n_main = 0;
    main.group_model = None;
    main.group_n = 0;
    main.group_scale_model = false;
    let q_main = one_block(&main, &th[..main_meta.theta_len], fixed_prec)?;

    let mut group = StructuredEffect::simple(&group_model, effect.group_n, group_meta.theta_len);
    group.scale_model = effect.group_scale_model;
    let q_group = one_block(&group, &th[main_meta.theta_len..], fixed_prec)?;

    // Latent order is group-major with the main index varying fastest.
    let q = kronecker_csc(&q_group, &q_main);
    if q.rows() != effect.n || q.cols() != effect.n {
        return Err(format!(
            "grouped effect {}: Q is {}x{}, expected {}x{}",
            effect.model_key(),
            q.rows(),
            q.cols(),
            effect.n,
            effect.n
        ));
    }
    Ok(q)
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

/// Fully validated structured layout shared by Q, prior, and constraint paths.
#[derive(Debug, Clone)]
pub struct StructuredPlan {
    pub effects: Vec<StructuredEffect>,
    pub latent_offsets: Vec<usize>,
    pub theta_offsets: Vec<usize>,
    pub latent_len: usize,
    pub theta_len: usize,
}

pub fn resolve_structured_plan(effects: &[StructuredEffect]) -> Result<StructuredPlan, String> {
    if effects.is_empty() {
        return Err("structured plan requires at least one effect".into());
    }
    let mut latent_offsets = Vec::with_capacity(effects.len());
    let mut theta_offsets = Vec::with_capacity(effects.len());
    let mut latent_len = 0usize;
    let mut theta_len = 0usize;
    for (i, effect) in effects.iter().enumerate() {
        let model = effect.model_key();
        if !SUPPORTED_MODELS.contains(&model.as_str()) {
            return Err(format!(
                "effect {model}: model is not executable by the shared structured path"
            ));
        }
        if effect.n == 0 {
            return Err(format!(
                "effect {i} ({model}): latent dimension must be positive"
            ));
        }
        let order = usize::try_from(effect.order.max(0)).unwrap_or(0);
        let meta = model_metadata(&model, order, effect.group_model.as_deref(), effect.cyclic)?;
        let allows_fixed_copy = model == "copy" && effect.theta_len == 0;
        if !allows_fixed_copy && effect.theta_len != meta.theta_len {
            return Err(format!(
                "effect {model}: declared theta_len {} != registry theta_len {}",
                effect.theta_len, meta.theta_len
            ));
        }
        if effect.group_model.is_some()
            && (effect.n_main == 0
                || effect.group_n == 0
                || effect.n_main.checked_mul(effect.group_n) != Some(effect.n))
        {
            return Err(format!(
                "effect {model}: n={} must equal n_main={} * group_n={}",
                effect.n, effect.n_main, effect.group_n
            ));
        }
        if model == "spde" {
            if effect.group_model.is_some() {
                return Err("grouped SPDE effects are not supported".into());
            }
            spde_fem(effect)?;
        }
        latent_offsets.push(latent_len);
        theta_offsets.push(theta_len);
        latent_len += effect.n;
        theta_len += effect.theta_len;
    }
    for (i, effect) in effects.iter().enumerate() {
        if effect.model_key() != "copy" {
            continue;
        }
        let source = effect
            .copy_of
            .ok_or_else(|| format!("copy effect {i}: missing source"))?;
        if source >= i {
            return Err(format!(
                "copy effect {i}: source {source} must appear first"
            ));
        }
        if effects[source].n != effect.n {
            return Err(format!(
                "copy effect {i}: n={} != source n={}",
                effect.n, effects[source].n
            ));
        }
    }
    Ok(StructuredPlan {
        effects: effects.to_vec(),
        latent_offsets,
        theta_offsets,
        latent_len,
        theta_len,
    })
}

/// Block-diagonal prior precision for concatenated θ across [`StructuredEffect`]s.
pub fn build_structured_precision(
    effects: &[StructuredEffect],
    theta: &[f64],
    fixed_prec: f64,
) -> Result<CscMatrix, String> {
    let plan = resolve_structured_plan(effects)?;
    let effects = plan.effects.as_slice();
    let expected = plan.theta_len;
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
        blocks.push(effect_block(effect, th, fixed_prec)?);
    }
    let q = block_diag_csc(&blocks)?;
    apply_copy_couplings(q, effects, theta)
}

/// Soft constraint \(x_{\mathrm{copy}} \approx \beta x_{\mathrm{src}}\):
/// \(Q_{\mathrm{src}} \mathrel{+}= \tau\beta^2 I\), \(Q_{\mathrm{cross}} = -\tau\beta I\).
/// The copy diagonal \(\tau I\) is already in the block from [`one_block`].
fn apply_copy_couplings(
    q: CscMatrix,
    effects: &[StructuredEffect],
    theta: &[f64],
) -> Result<CscMatrix, String> {
    let mut offsets = Vec::with_capacity(effects.len());
    let mut off = 0usize;
    for e in effects {
        offsets.push(off);
        off += e.n;
    }
    let n = off;
    let mut theta_off = 0usize;
    let mut theta_offs = Vec::with_capacity(effects.len());
    for e in effects {
        theta_offs.push(theta_off);
        theta_off += e.theta_len;
    }

    let mut extra: Vec<(usize, usize, f64)> = Vec::new();
    let tau = COPY_PRECISION;
    for (i, e) in effects.iter().enumerate() {
        if e.model_key() != "copy" {
            continue;
        }
        let src = e.copy_of.ok_or_else(|| {
            format!("copy effect {i}: missing copy_of (source must appear first)")
        })?;
        if src >= effects.len() || src == i {
            return Err(format!("copy effect {i}: invalid source index {src}"));
        }
        if src > i {
            return Err(format!(
                "copy effect {i}: source {src} must appear before the copy"
            ));
        }
        if effects[src].n != e.n {
            return Err(format!(
                "copy effect {i}: n={} != source n={}",
                e.n, effects[src].n
            ));
        }
        let beta = if e.theta_len > 0 {
            theta
                .get(theta_offs[i])
                .copied()
                .ok_or_else(|| format!("copy effect {i}: missing beta in theta"))?
        } else {
            1.0
        };
        if !beta.is_finite() {
            return Err(format!("copy effect {i}: non-finite beta"));
        }
        let src_off = offsets[src];
        let cpy_off = offsets[i];
        let tau_b2 = tau * beta * beta;
        let tau_b = tau * beta;
        for k in 0..e.n {
            extra.push((src_off + k, src_off + k, tau_b2));
            extra.push((src_off + k, cpy_off + k, -tau_b));
            extra.push((cpy_off + k, src_off + k, -tau_b));
        }
    }
    if extra.is_empty() {
        return Ok(q);
    }
    let extra_q = sparse_from_triplets(n, n, &extra);
    add_csc(&q, &extra_q)
}

/// Default hyperprior stack, validated against each effect's declared θ block.
pub fn structured_prior_stack(effects: &[StructuredEffect]) -> Result<HyperPriorStack, String> {
    resolve_structured_plan(effects)?;
    let mut priors = Vec::new();
    for effect in effects {
        let m = effect.model_key();
        if m == "fixed" || (m == "copy" && effect.theta_len == 0) {
            continue;
        }
        let order = if matches!(m.as_str(), "ar" | "arp") {
            usize::try_from(effect.order.max(0)).unwrap_or(0)
        } else {
            0
        };
        let mut stack = HyperPriorStack::default_for_effect_order(&m, order)?;
        if let Some(group_model) = effect.group_model.as_deref() {
            let group_stack = HyperPriorStack::default_for_effect_order(group_model, 0)?;
            stack.priors.extend(group_stack.priors);
        }
        if stack.theta_dim() != effect.theta_len {
            return Err(format!(
                "effect {m}: prior theta dimension {} != declared theta_len {}",
                stack.theta_dim(),
                effect.theta_len
            ));
        }
        priors.extend(stack.priors);
    }
    Ok(HyperPriorStack::new(priors))
}

fn component_constraints(adj: &[Vec<usize>]) -> Result<Option<ConstraintSpec>, String> {
    let components = graph_components(adj)?;
    let constrained = components
        .iter()
        .filter(|component| component.len() > 1)
        .collect::<Vec<_>>();
    if constrained.is_empty() {
        return Ok(None);
    }
    let n = adj.len();
    let k = constrained.len();
    let mut a = vec![0.0; k * n];
    for (r, component) in constrained.into_iter().enumerate() {
        let weight = 1.0 / (component.len() as f64).sqrt();
        for &node in component {
            a[r * n + node] = weight;
        }
    }
    Ok(Some(ConstraintSpec {
        n,
        k,
        a,
        e: vec![0.0; k],
        method: ConstraintMethod::Augmented,
    }))
}

fn crw2_constraints(effect: &StructuredEffect) -> Result<ConstraintSpec, String> {
    let layout = if effect.crw2_layout.is_empty() {
        "simple"
    } else {
        effect.crw2_layout.as_str()
    };
    let n_knots = if layout == "simple" {
        effect.n
    } else {
        if !effect.n.is_multiple_of(2) {
            return Err(format!(
                "crw2 layout '{layout}' requires an even latent size"
            ));
        }
        effect.n / 2
    };
    if n_knots < 3 {
        return Err("crw2 requires at least three knot locations".into());
    }
    let positions = effect
        .positions
        .clone()
        .unwrap_or_else(|| (0..n_knots).map(|i| i as f64).collect());
    if positions.len() != n_knots {
        return Err(format!(
            "crw2 layout '{layout}': positions length {} != knot count {n_knots}",
            positions.len()
        ));
    }
    let mean = positions.iter().sum::<f64>() / n_knots as f64;
    let mut a = vec![0.0; 2 * effect.n];
    let constant_norm = (n_knots as f64).sqrt();
    let mut trend_norm_sq = 0.0;
    match layout {
        "simple" => {
            for (i, &t) in positions.iter().enumerate() {
                a[i] = 1.0 / constant_norm;
                a[effect.n + i] = t - mean;
                trend_norm_sq += (t - mean).powi(2);
            }
        }
        "pairs" => {
            for (i, &t) in positions.iter().enumerate() {
                a[2 * i] = 1.0 / constant_norm;
                a[effect.n + 2 * i] = t - mean;
                a[effect.n + 2 * i + 1] = 1.0;
                trend_norm_sq += (t - mean).powi(2) + 1.0;
            }
        }
        "block" => {
            for (i, &t) in positions.iter().enumerate() {
                a[i] = 1.0 / constant_norm;
                a[effect.n + i] = t - mean;
                a[effect.n + n_knots + i] = 1.0;
                trend_norm_sq += (t - mean).powi(2) + 1.0;
            }
        }
        other => return Err(format!("unknown crw2 layout '{other}'")),
    }
    if trend_norm_sq <= 0.0 {
        return Err("crw2 positions do not define a linear trend".into());
    }
    let trend_norm = trend_norm_sq.sqrt();
    for value in &mut a[effect.n..] {
        *value /= trend_norm;
    }
    Ok(ConstraintSpec {
        n: effect.n,
        k: 2,
        a,
        e: vec![0.0; 2],
        method: ConstraintMethod::Augmented,
    })
}

fn grouped_constraints(effect: &StructuredEffect) -> Result<Option<ConstraintSpec>, String> {
    let Some(group_model) = effect.group_model.as_deref() else {
        return Ok(None);
    };
    if effect.n_main == 0 || effect.group_n == 0 || effect.n != effect.n_main * effect.group_n {
        return Err(format!(
            "grouped effect {}: invalid main/group dimensions",
            effect.model_key()
        ));
    }
    let order = usize::try_from(effect.order.max(0)).unwrap_or(0);
    let main_meta = model_metadata(&effect.model_key(), order, None, effect.cyclic)?;
    let group_meta = model_metadata(group_model, 0, None, false)?;
    let mut main = effect.clone();
    main.n = effect.n_main;
    main.theta_len = main_meta.theta_len;
    main.n_main = 0;
    main.group_model = None;
    main.group_n = 0;
    main.group_scale_model = false;
    let mut group = StructuredEffect::simple(group_model, effect.group_n, group_meta.theta_len);
    group.scale_model = effect.group_scale_model;

    let main_c = structured_constraints(std::slice::from_ref(&main))?;
    let group_c = structured_constraints(std::slice::from_ref(&group))?;
    if main_c.is_none() && group_c.is_none() {
        return Ok(None);
    }

    let n = effect.n;
    let mut candidates: Vec<Vec<f64>> = Vec::new();
    if let Some(c) = main_c {
        for g in 0..effect.group_n {
            for r in 0..c.k {
                let mut row = vec![0.0; n];
                let src = &c.a[r * effect.n_main..(r + 1) * effect.n_main];
                row[g * effect.n_main..(g + 1) * effect.n_main].copy_from_slice(src);
                candidates.push(row);
            }
        }
    }
    if let Some(c) = group_c {
        for main_i in 0..effect.n_main {
            for r in 0..c.k {
                let mut row = vec![0.0; n];
                for g in 0..effect.group_n {
                    row[g * effect.n_main + main_i] = c.a[r * effect.group_n + g];
                }
                candidates.push(row);
            }
        }
    }

    // Remove the k_main*k_group duplicated intersection and normalize the basis.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for mut row in candidates {
        for prior in &basis {
            let dot = row.iter().zip(prior).map(|(a, b)| a * b).sum::<f64>();
            for (value, p) in row.iter_mut().zip(prior) {
                *value -= dot * p;
            }
        }
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for value in &mut row {
                *value /= norm;
            }
            basis.push(row);
        }
    }
    let k = basis.len();
    let a = basis.into_iter().flatten().collect();
    Ok(Some(ConstraintSpec {
        n,
        k,
        a,
        e: vec![0.0; k],
        method: ConstraintMethod::Augmented,
    }))
}

/// Hard linear constraints for intrinsic / BYM / rw2d blocks.
pub fn structured_constraints(
    effects: &[StructuredEffect],
) -> Result<Option<ConstraintSpec>, String> {
    resolve_structured_plan(effects)?;
    let full_n: usize = effects.iter().map(|e| e.n).sum();
    let mut stacked: Option<ConstraintSpec> = None;
    let mut offset = 0usize;
    for effect in effects {
        let typ = effect.model_key();
        let n_e = effect.n;
        if effect.group_model.is_some() {
            if let Some(block) = grouped_constraints(effect)? {
                let embedded = block.embed(full_n, offset)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded)?,
                });
            }
            offset += n_e;
            continue;
        }
        if typ == "besag" {
            let adj = effect
                .adj
                .as_deref()
                .ok_or_else(|| "besag missing adj".to_string())?;
            if let Some(block) = component_constraints(adj)? {
                let embedded = block.embed(full_n, offset)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded)?,
                });
            }
            offset += n_e;
            continue;
        }
        if typ == "crw2" {
            let block = crw2_constraints(effect)?;
            let embedded = block.embed(full_n, offset)?;
            stacked = Some(match stacked {
                None => embedded,
                Some(prev) => prev.vstack(&embedded)?,
            });
            offset += n_e;
            continue;
        }
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
            let season = effect.season_len().max(2);
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
            let adj = effect
                .adj
                .as_deref()
                .ok_or_else(|| "bym missing adj".to_string())?;
            if let Some(block) = component_constraints(adj)? {
                let embedded = block.embed(full_n, offset)?;
                stacked = Some(match stacked {
                    None => embedded,
                    Some(prev) => prev.vstack(&embedded)?,
                });
            }
            offset += n_e;
            continue;
        }
        if typ == "bym2" {
            // The current combined-field BYM2 precision is proper for φ < 1.
            offset += n_e;
            continue;
        }
        if typ == "rw2" {
            // Match R-INLA `constr=TRUE`: sum-to-zero only. Rank deficiency 2
            // is handled by `RW2_CONSTR_DIAGONAL` on Q, not a linear extraconstr.
            let block = sum_to_zero_constraint(n_e, 1)?;
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
    use crate::registry::{SUPPORTED_MODELS, model_metadata};

    fn representative_effect(model: &str) -> StructuredEffect {
        let order = if matches!(model, "ar" | "arp") { 2 } else { 0 };
        let meta = model_metadata(model, order, None, false).unwrap();
        let n = match model {
            "rw2d" | "matern2d" => 9,
            "bym" => 8,
            "iid2d" => 8,
            "iid3d" => 9,
            "iid4d" => 8,
            "iid5d" => 10,
            _ => 8,
        };
        let mut effect = StructuredEffect::simple(model, n, meta.theta_len);
        effect.order = order as i32;
        match model {
            "rw2d" | "matern2d" => {
                effect.nrow = 3;
                effect.ncol = 3;
            }
            "besag" | "bym2" => {
                effect.adj = Some(vec![vec![1], vec![0, 2], vec![1, 3], vec![2]]);
                effect.n = 4;
            }
            "bym" => {
                effect.adj = Some(vec![vec![1], vec![0, 2], vec![1, 3], vec![2]]);
            }
            "seasonal" => effect.season = 4,
            "crw1" | "crw2" => {
                effect.positions = Some((0..n).map(|i| i as f64).collect());
            }
            "spde" => {
                effect.n = 4;
                effect.mesh = Some(Box::new(SpdeMesh::isotropic_2d(
                    vec![(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
                    vec![[0, 2, 1], [1, 2, 3]],
                )));
            }
            _ => {}
        }
        effect
    }

    #[test]
    fn every_advertised_structured_model_has_a_complete_contract() {
        for model in SUPPORTED_MODELS {
            if *model == "copy" {
                continue;
            }
            let effect = representative_effect(model);
            let order = usize::try_from(effect.order.max(0)).unwrap_or(0);
            let meta = model_metadata(model, order, None, effect.cyclic).unwrap();
            let stack = structured_prior_stack(std::slice::from_ref(&effect))
                .unwrap_or_else(|e| panic!("model {model}: prior contract: {e}"));
            assert_eq!(stack.theta_dim(), meta.theta_len, "model {model}");
            let q = build_structured_precision(
                std::slice::from_ref(&effect),
                &meta.default_theta,
                1e-4,
            )
            .unwrap_or_else(|e| panic!("model {model}: Q contract: {e}"));
            assert_eq!(q.rows(), effect.n, "model {model}: Q rows");
            assert_eq!(q.cols(), effect.n, "model {model}: Q cols");
            assert!(
                q.data().iter().all(|v| v.is_finite()),
                "model {model}: non-finite Q"
            );
        }
    }

    #[test]
    fn unsupported_metadata_only_models_are_rejected_by_structured_path() {
        for model in ["besag2", "rgeneric"] {
            let meta = model_metadata(model, 0, None, false).unwrap();
            let effect = StructuredEffect::simple(model, 8, meta.theta_len);
            let err = build_structured_precision(&[effect], &meta.default_theta, 1e-4).unwrap_err();
            assert!(err.contains("not executable"), "model {model}: {err}");
        }
    }

    #[test]
    fn spde_without_mesh_is_rejected() {
        let meta = model_metadata("spde", 0, None, false).unwrap();
        let effect = StructuredEffect::simple("spde", 4, meta.theta_len);
        let err = build_structured_precision(&[effect], &meta.default_theta, 1e-4).unwrap_err();
        assert!(
            err.contains("spde missing") || err.contains("no vertices"),
            "{err}"
        );
    }

    #[test]
    fn spde_1d_mesh_q_matches_knot_count() {
        let meta = model_metadata("spde", 0, None, false).unwrap();
        let loc = vec![0.0, 1.0, 2.0, 4.0];
        let mut effect = StructuredEffect::simple("spde", loc.len(), meta.theta_len);
        effect.mesh = Some(Box::new(SpdeMesh::knots_1d(loc.clone())));
        let q =
            build_structured_precision(std::slice::from_ref(&effect), &meta.default_theta, 1e-4)
                .unwrap();
        assert_eq!(q.rows(), loc.len());
        assert!(q.data().iter().all(|v| v.is_finite()));
    }

    fn lattice_2d(nx: usize, ny: usize) -> (Vec<(f64, f64)>, Vec<[usize; 3]>) {
        let mut vertices = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                vertices.push((i as f64 / (nx - 1) as f64, j as f64 / (ny - 1) as f64));
            }
        }
        let idx = |i: usize, j: usize| j * nx + i;
        let mut triangles = Vec::new();
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v01 = idx(i, j + 1);
                let v11 = idx(i + 1, j + 1);
                triangles.push([v00, v10, v01]);
                triangles.push([v10, v11, v01]);
            }
        }
        (vertices, triangles)
    }

    #[test]
    fn barrier_spde_reduces_covariance_across_strip() {
        let meta = model_metadata("spde", 0, None, false).unwrap();
        let (vertices, triangles) = lattice_2d(7, 2);
        let barrier: Vec<usize> = triangles
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                let cx = (vertices[t[0]].0 + vertices[t[1]].0 + vertices[t[2]].0) / 3.0;
                (0.35..0.65).contains(&cx)
            })
            .map(|(k, _)| k)
            .collect();
        assert!(!barrier.is_empty());

        let mut open = StructuredEffect::simple("spde", vertices.len(), meta.theta_len);
        open.mesh = Some(Box::new(SpdeMesh::isotropic_2d(
            vertices.clone(),
            triangles.clone(),
        )));
        let mut blocked = open.clone();
        if let Some(mesh) = blocked.mesh.as_mut() {
            mesh.barrier_triangles = barrier;
            mesh.range_fraction = 0.1;
        }
        let q_open =
            build_structured_precision(std::slice::from_ref(&open), &meta.default_theta, 1e-4)
                .unwrap();
        let q_bar =
            build_structured_precision(std::slice::from_ref(&blocked), &meta.default_theta, 1e-4)
                .unwrap();
        let n = vertices.len();
        let cov_open =
            inla_math::invert_symmetric_matrix(&inla_math::csc_to_dense(&q_open).unwrap(), n)
                .unwrap();
        let cov_bar =
            inla_math::invert_symmetric_matrix(&inla_math::csc_to_dense(&q_bar).unwrap(), n)
                .unwrap();
        let left = 0;
        let right = 6;
        let open_lr = cov_open[left * n + right].abs();
        let bar_lr = cov_bar[left * n + right].abs();
        assert!(
            bar_lr < 0.5 * open_lr,
            "barrier should cut cross-strip covariance: open={open_lr} barrier={bar_lr}"
        );
    }

    #[test]
    fn anisotropic_spde_q_differs_from_isotropic() {
        let meta = model_metadata("spde", 0, None, false).unwrap();
        let (vertices, triangles) = lattice_2d(3, 3);
        let mut iso = StructuredEffect::simple("spde", vertices.len(), meta.theta_len);
        iso.mesh = Some(Box::new(SpdeMesh::isotropic_2d(
            vertices.clone(),
            triangles.clone(),
        )));
        let mut aniso = iso.clone();
        if let Some(mesh) = aniso.mesh.as_mut() {
            mesh.diffusion = [3.0, 0.5, 0.4];
        }
        let q_iso =
            build_structured_precision(std::slice::from_ref(&iso), &meta.default_theta, 1e-4)
                .unwrap();
        let q_an =
            build_structured_precision(std::slice::from_ref(&aniso), &meta.default_theta, 1e-4)
                .unwrap();
        assert_ne!(q_iso.data(), q_an.data());
    }

    #[test]
    fn ar1_block_precision_finite() {
        let effects = [StructuredEffect::simple("ar1", 8, 2)];
        let q = build_structured_precision(&effects, &[0.0, 0.0], 1e-4).unwrap();
        assert_eq!(q.rows(), 8);
        assert!(q.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn grouped_iid_ar1_uses_group_major_kronecker_order() {
        let mut effect = StructuredEffect::simple("iid", 12, 3);
        effect.n_main = 4;
        effect.group_model = Some("ar1".into());
        effect.group_n = 3;
        let theta = [0.2, -0.1, 0.6];
        let q = build_structured_precision(std::slice::from_ref(&effect), &theta, 1e-4).unwrap();
        let q_main = iid_precision_csc(4, theta[0].exp()).unwrap();
        let rho = 2.0 / (1.0 + (-theta[2]).exp()) - 1.0;
        let q_group = ar1_precision_csc(3, rho, theta[1].exp()).unwrap();
        let expected = kronecker_csc(&q_group, &q_main);
        assert_eq!(q.to_dense(), expected.to_dense());
        assert_eq!(
            structured_prior_stack(std::slice::from_ref(&effect))
                .unwrap()
                .theta_dim(),
            3
        );
        assert!(structured_constraints(&[effect]).unwrap().is_none());
    }

    #[test]
    fn grouped_intrinsic_constraints_remove_redundant_intersection() {
        let mut effect = StructuredEffect::simple("rw1", 6, 2);
        effect.n_main = 3;
        effect.group_model = Some("rw1".into());
        effect.group_n = 2;
        let q = build_structured_precision(std::slice::from_ref(&effect), &[0.0, 0.0], 1e-4)
            .unwrap()
            .to_dense();
        let c = structured_constraints(&[effect]).unwrap().unwrap();
        assert_eq!(c.k, 4); // 2*1 + 3*1 - 1*1
        for r in 0..c.k {
            let row = &c.a[r * c.n..(r + 1) * c.n];
            for i in 0..c.n {
                let value = (0..c.n).map(|j| q[[i, j]] * row[j]).sum::<f64>();
                assert!(value.abs() < 1e-8, "row {r}, i {i}: {value}");
            }
        }
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
    fn iid2d_block_length_and_nnz() {
        let effects = [StructuredEffect::simple("iid2d", 8, 3)];
        let q = build_structured_precision(&effects, &[0.0, 0.0, 0.0], 1e-4).unwrap();
        assert_eq!(q.rows(), 8);
        // 4 units × 2×2 block, uncorrelated ⇒ 8 diagonal entries
        assert_eq!(q.nnz(), 8);
        let stack = structured_prior_stack(&effects).unwrap();
        assert_eq!(stack.theta_dim(), 3);
        assert!(stack.log_density(&[0.0, 0.0, 0.0]).unwrap().is_finite());
    }

    #[test]
    fn seasonal_constraints_kill_the_whole_null_space() {
        let season = 4usize;
        let n = 24usize;
        let mut effect = StructuredEffect::simple("seasonal", n, 1);
        effect.season = season;

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
    fn rw2_constraints_match_rinla_sum_to_zero_only() {
        let effects = [StructuredEffect::simple("rw2", 6, 1)];
        let c = structured_constraints(&effects).unwrap().unwrap();
        assert_eq!(c.k, 1);
        assert_eq!(c.n, 6);
    }

    #[test]
    fn besag_constraints_follow_connected_components() {
        let adj = vec![vec![1], vec![0, 2], vec![1], vec![4], vec![3], vec![]];
        let mut effect = StructuredEffect::simple("besag", adj.len(), 1);
        effect.adj = Some(adj);
        let c = structured_constraints(&[effect]).unwrap().unwrap();
        assert_eq!(c.k, 2);
        assert_eq!(c.n, 6);
        assert_eq!(c.a[5], 0.0, "singleton must not be constrained");
        assert_eq!(c.a[6 + 5], 0.0, "singleton must not be constrained");
        assert!(c.a[..3].iter().all(|v| *v > 0.0));
        assert!(c.a[6 + 3..6 + 5].iter().all(|v| *v > 0.0));
    }

    #[test]
    fn crw2_constraints_span_the_true_null_space_for_all_layouts() {
        let positions = vec![0.0, 0.7, 2.0, 4.5, 8.0];
        for layout in ["simple", "pairs", "block"] {
            let n = if layout == "simple" {
                positions.len()
            } else {
                2 * positions.len()
            };
            let mut effect = StructuredEffect::simple("crw2", n, 1);
            effect.positions = Some(positions.clone());
            effect.crw2_layout = layout.into();
            let c = structured_constraints(std::slice::from_ref(&effect))
                .unwrap()
                .unwrap();
            assert_eq!(c.k, 2, "layout {layout}");
            let q = build_structured_precision(&[effect], &[0.0], 1e-4)
                .unwrap()
                .to_dense();
            for r in 0..2 {
                let row = &c.a[r * n..(r + 1) * n];
                for i in 0..n {
                    let value = (0..n).map(|j| q[[i, j]] * row[j]).sum::<f64>();
                    assert!(
                        value.abs() < 1e-8,
                        "layout {layout}, row {r}, i {i}: {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn rw2_unit_positions_match_equal_spacing_q() {
        let n = 8usize;
        let pos: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut with_pos = StructuredEffect::simple("rw2", n, 1);
        with_pos.positions = Some(pos);
        let q_pos = build_structured_precision(&[with_pos], &[0.0], 1e-4).unwrap();
        let q_eq =
            build_structured_precision(&[StructuredEffect::simple("rw2", n, 1)], &[0.0], 1e-4)
                .unwrap();
        let d_pos = q_pos.to_dense();
        let d_eq = q_eq.to_dense();
        for i in 0..n {
            for j in 0..n {
                let a = d_pos[[i, j]];
                let b = d_eq[[i, j]];
                assert!((a - b).abs() < 1e-10, "Q[{i},{j}] {a} vs {b}");
            }
        }
    }

    #[test]
    fn rw2_irregular_positions_change_q() {
        let n = 6usize;
        let mut irregular = StructuredEffect::simple("rw2", n, 1);
        irregular.positions = Some(vec![0.0, 1.0, 1.5, 4.0, 8.0, 9.0]);
        let q_irr = build_structured_precision(&[irregular], &[0.0], 1e-4).unwrap();
        let q_eq =
            build_structured_precision(&[StructuredEffect::simple("rw2", n, 1)], &[0.0], 1e-4)
                .unwrap();
        let d_irr = q_irr.to_dense();
        let d_eq = q_eq.to_dense();
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_diff = max_diff.max((d_irr[[i, j]] - d_eq[[i, j]]).abs());
            }
        }
        assert!(
            max_diff > 0.1,
            "irregular RW2 Q should differ from equal-spacing, max_diff={max_diff}"
        );
    }

    #[test]
    fn rw2_galerkin_kernel_includes_linear_in_locations() {
        let pos = [0.0, 1.0, 2.5, 4.0, 7.0, 9.0];
        let n = pos.len();
        let q = crw2_precision_csc(&pos, 1.0, "simple").unwrap();
        let dense = q.to_dense();
        let mut q1 = 0.0_f64;
        let mut qp = 0.0_f64;
        for i in 0..n {
            let mut s1 = 0.0;
            let mut sp = 0.0;
            for j in 0..n {
                s1 += dense[[i, j]];
                sp += dense[[i, j]] * pos[j];
            }
            q1 += s1.abs();
            qp += sp.abs();
        }
        assert!(q1 < 1e-8, "Q 1 = {q1}");
        assert!(qp < 1e-8, "Q pos = {qp}");
    }

    #[test]
    fn copy_couples_source_with_beta() {
        let mut copy = StructuredEffect::simple("copy", 3, 1);
        copy.copy_of = Some(0);
        let effects = [StructuredEffect::simple("iid", 3, 1), copy];
        let beta = 1.5;
        let q = build_structured_precision(&effects, &[0.0, beta], 1e-4).unwrap();
        let dense = q.to_dense();
        let tau = COPY_PRECISION;
        // iid τ=exp(0)=1, plus τ_copy β² on the source diagonal
        assert!((dense[[0, 0]] - (1.0 + tau * beta * beta)).abs() < 1e-6);
        assert!((dense[[3, 3]] - tau).abs() < 1e-6);
        assert!((dense[[0, 3]] + tau * beta).abs() < 1e-6);
        assert!((dense[[3, 0]] + tau * beta).abs() < 1e-6);
        assert_eq!(structured_prior_stack(&effects).unwrap().theta_dim(), 2);
        // copy has no hard constraints
        assert!(structured_constraints(&effects).unwrap().is_none());
    }

    #[test]
    fn q_block_must_match_declared_latent_size() {
        let mut effect = StructuredEffect::simple("rw2d", 8, 1);
        effect.nrow = 3;
        effect.ncol = 3;
        let err =
            build_structured_precision(std::slice::from_ref(&effect), &[0.0], 1e-4).unwrap_err();
        assert!(err.contains("Q is") || err.contains("expected"), "{err}");
    }
}

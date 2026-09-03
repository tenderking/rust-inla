import numpy as np
import pytest

import inla
from inla import _native as core


def _make_graph_19():
    """Synthetic 19-region connected spatial graph."""
    adj = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 4, 5],
        3: [1, 6],
        4: [2, 5, 7],
        5: [2, 4, 8, 9],
        6: [3, 10],
        7: [4, 8, 11],
        8: [5, 7, 9, 12],
        9: [5, 8, 13],
        10: [6, 14],
        11: [7, 12, 15],
        12: [8, 11, 13, 16],
        13: [9, 12, 17],
        14: [10, 18],
        15: [11, 16],
        16: [12, 15, 17, 18],
        17: [13, 16],
        18: [14, 16],
    }
    return [adj[i] for i in range(19)]


def test_free_gaussian_precision():
    """Verify free Gaussian observation precision is estimated jointly with latent field."""
    rng = np.random.default_rng(42)
    n = 50
    true_tau_obs = 16.0  # sd = 0.25
    true_tau_rw1 = 4.0

    # Simulate RW1 + Gaussian noise
    rw1 = np.cumsum(rng.normal(0, 1.0 / np.sqrt(true_tau_rw1), size=n))
    rw1 -= rw1.mean()
    y = rw1 + rng.normal(0, 1.0 / np.sqrt(true_tau_obs), size=n)

    data = {"y": y, "time": np.arange(n)}
    res = inla("y ~ -1 + f(time, model='rw1')", data=data, family="gaussian")

    # Mode contains: 1. Precision for Gaussian observations, 2. Precision for time
    assert len(res.mode) == 2
    assert "Precision for the Gaussian observations" in res.summary_hyperpar["names"]
    assert "Precision for time" in res.summary_hyperpar["names"]

    # Estimated observation precision should be close to true value ~16
    est_tau_obs = float(res.summary_hyperpar["mean"][0])
    assert 5.0 < est_tau_obs < 40.0

    # Ensure $ID is attached to summary_random
    assert "time" in res.summary_random
    assert "ID" in res.summary_random["time"]
    assert np.array_equal(res.summary_random["time"]["ID"], np.arange(n))


def test_float_group_indices_and_id():
    """Verify float indices (from inla.group) are not truncated to int and retain exact IDs."""
    rng = np.random.default_rng(123)
    # Floating point bin medians, e.g. from inla.group(x, n=20)
    group_medians = np.linspace(0.123, 9.876, 20)
    covariate = np.repeat(group_medians, 3)
    n = len(covariate)

    true_effect = np.sin(group_medians)
    f_map = {m: true_effect[i] for i, m in enumerate(group_medians)}
    y_latent = np.array([f_map[c] for c in covariate])
    y = y_latent + rng.normal(0, 0.1, size=n)

    data = {"y": y, "cov_group": covariate}
    res = inla("y ~ -1 + f(cov_group, model='rw2')", data=data, family="gaussian")

    assert "cov_group" in res.summary_random
    tab = res.summary_random["cov_group"]
    assert "ID" in tab
    # ID should match the unique sorted float levels exactly
    np.testing.assert_allclose(tab["ID"], group_medians, atol=1e-12)
    assert len(tab["mean"]) == len(group_medians)


def test_rw2_infers_positions_from_group_ids():
    """Classic rw2/rw1: irregular numeric IDs are knot positions (Lindgren & Rue 2008)."""
    from inla.api import _effect_positions

    ids = np.array([0.0, 1.0, 1.5, 8.0, 9.0])
    np.testing.assert_allclose(
        _effect_positions("rw2", None, ids=ids, n_knots=5),
        ids,
    )
    np.testing.assert_allclose(
        _effect_positions("rw1", None, ids=ids, n_knots=5),
        ids,
    )
    np.testing.assert_allclose(
        _effect_positions("crw2", None, ids=ids, n_knots=5),
        ids,
    )
    np.testing.assert_allclose(
        _effect_positions("rw2", ids.tolist(), n_knots=5),
        ids,
    )
    # Cyclic rw1/rw2 skips location inference
    assert _effect_positions("rw2", None, ids=ids, n_knots=5, cyclic=True) is None
    assert _effect_positions("rw1", None, ids=ids, n_knots=5, cyclic=True) is None

    # Numeric vector of bad length raises with n_knots
    with pytest.raises(ValueError, match="expected 5 knots"):
        _effect_positions("rw2", [0.0, 1.0], n_knots=5)

    # Observation-length column maps in knot/ids order (Nit 1)
    knot_ids = np.array([10, 20, 30])
    obs_zcol = np.array([2, 0, 1, 0, 2, 1])
    obs_pos = np.array([300.0, 100.0, 200.0, 100.0, 300.0, 200.0])
    mapped = _effect_positions(
        "rw2",
        "pcol",
        data={"pcol": obs_pos},
        ids=knot_ids,
        zcol=obs_zcol,
        n_obs=6,
        n_knots=3,
    )
    np.testing.assert_allclose(mapped, [100.0, 200.0, 300.0])


def test_rw2_equal_spacing_q_differs_from_irregular_medians():
    ids = [0.0, 1.0, 1.5, 8.0, 9.0]
    q_eq = core.rw2_precision_matrix(5).to_scipy().toarray()
    q_gal = core.crw2_precision_matrix(ids, 1.0, layout="simple").to_scipy().toarray()
    assert float(np.max(np.abs(q_eq - q_gal))) > 1.0


def test_besag_and_rw2_joint_model():
    """Verify joint Besag + RW2 model with free Gaussian precision reproduces latent fields."""
    rng = np.random.default_rng(999)
    adj = _make_graph_19()
    n_spatial = 19
    n_bins = 30
    n_obs = 150

    # Spatial indices (1-based for Besag) and continuous covariate binned into float medians
    spatial_id = rng.integers(1, n_spatial + 1, size=n_obs)
    cov_bins = np.linspace(10.5, 85.5, n_bins)
    cov_id = rng.choice(cov_bins, size=n_obs)

    # True spatial field on graph
    # Smooth spatial pattern across graph
    true_spatial = np.sin(np.arange(n_spatial) * 0.5)
    true_spatial -= true_spatial.mean()

    # True non-linear effect for RW2
    true_rw2 = np.cos(np.linspace(0, 2 * np.pi, n_bins))
    true_rw2 -= true_rw2.mean()
    rw2_map = {b: true_rw2[i] for i, b in enumerate(cov_bins)}

    true_tau_obs = 25.0  # sd = 0.2
    y_signal = np.array([true_spatial[s - 1] + rw2_map[c] for s, c in zip(spatial_id, cov_id)])
    y = y_signal + rng.normal(0, 1.0 / np.sqrt(true_tau_obs), size=n_obs)

    data = {
        "y": y,
        "region": spatial_id,
        "cov": cov_id,
        "adj_list": adj,
    }

    res = inla(
        "y ~ 1 + f(region, model='besag', scale_model=True) + f(cov, model='rw2', scale_model=True)",
        data=data,
        family="gaussian",
        deterministic=True,
    )

    # Hyperparameters: 1. Gaussian prec, 2. Besag prec, 3. RW2 prec
    assert len(res.mode) == 3
    assert len(res.summary_hyperpar["names"]) == 3
    assert res.summary_hyperpar["names"][0] == "Precision for the Gaussian observations"
    assert res.summary_hyperpar["names"][1] == "Precision for region"
    assert res.summary_hyperpar["names"][2] == "Precision for cov"

    # Latent effects correlation with ground truth
    besag_est = res.summary_random["region"]["mean"]
    rw2_est = res.summary_random["cov"]["mean"]

    corr_besag = np.corrcoef(besag_est, true_spatial)[0, 1]
    corr_rw2 = np.corrcoef(rw2_est, true_rw2)[0, 1]

    # Besag and RW2 should both be strongly correlated with true signals
    assert corr_besag > 0.90, f"Besag correlation {corr_besag} too low"
    assert corr_rw2 > 0.90, f"RW2 correlation {corr_rw2} too low"

    # Verify IDs
    np.testing.assert_array_equal(res.summary_random["region"]["ID"], np.arange(1, n_spatial + 1))
    np.testing.assert_allclose(res.summary_random["cov"]["ID"], cov_bins, atol=1e-12)


def test_scale_model_csc_parity():
    """Moore–Penrose scale.model factors for Besag/equal-spacing RW2.

    These are the geometric-mean-of-diag(ginv(Q)) factors used by
    ``scale_model_csc``. Equal-spacing RW2 of length 25/73 is the case
    R-INLA ``inla.rw(n, order=2, scale.model=TRUE)`` targets.
    """
    # n=6 cycle: exact scale factor is 72/35 ~= 2.057142857
    adj_6 = [[1, 5], [0, 2], [1, 3], [2, 4], [3, 5], [4, 0]]
    q_6 = core.besag_precision_matrix(adj_6)
    q_scaled_6 = core.scale_model_csc(q_6)
    diag_6 = np.diag(q_scaled_6.to_scipy().toarray())
    np.testing.assert_allclose(diag_6, np.full(6, 35.0 / 36.0), rtol=1e-6)

    adj_19 = _make_graph_19()
    q_19 = core.besag_precision_matrix(adj_19)
    q_scaled_19 = core.scale_model_csc(q_19)
    fac_19 = q_scaled_19.to_scipy().toarray()[0, 0] / q_19.to_scipy().toarray()[0, 0]
    np.testing.assert_allclose(fac_19, 0.657130019494448, rtol=1e-6)

    q_rw2_25 = core.rw2_precision_matrix(25)
    q_rw2_scaled_25 = core.scale_model_csc(q_rw2_25)
    fac_25 = q_rw2_scaled_25.to_scipy().toarray()[2, 2] / q_rw2_25.to_scipy().toarray()[2, 2]
    np.testing.assert_allclose(fac_25, 26.988486337199, rtol=1e-6)

    q_rw2_73 = core.rw2_precision_matrix(73)
    q_rw2_scaled_73 = core.scale_model_csc(q_rw2_73)
    fac_73 = q_rw2_scaled_73.to_scipy().toarray()[2, 2] / q_rw2_73.to_scipy().toarray()[2, 2]
    np.testing.assert_allclose(fac_73, 666.76857619175, rtol=1e-6)


def test_rw2_unit_spacing_galerkin_matches_d2():
    """R-INLA irregular RW2 (Galerkin / crw2 simple) reduces to D₂'D₂ at h=1."""
    n = 12
    pos = list(range(n))
    q_d2 = core.rw2_precision_matrix(n, 1.0).to_scipy().toarray()
    q_gal = core.crw2_precision_matrix(pos, 1.0, layout="simple").to_scipy().toarray()
    np.testing.assert_allclose(q_gal, q_d2, atol=1e-10)


def test_locked_theta_operator_parity():
    """Q blocks, scale.model, and R-INLA-style sum-to-zero at a locked τ."""
    rng = np.random.default_rng(777)
    adj = _make_graph_19()
    n_spatial = 19
    n_obs = 100

    true_u = np.sin(np.arange(n_spatial) * 0.5)
    true_u -= true_u.mean()

    region_id = rng.integers(1, n_spatial + 1, size=n_obs)
    u_obs = np.array([true_u[r - 1] for r in region_id])
    z_raw = u_obs + rng.normal(0, 0.1, size=n_obs)

    z_bins = np.sort(np.unique(np.round(z_raw, 1)))
    n_rw2 = len(z_bins)
    z_binned = z_bins[np.abs(z_raw[:, None] - z_bins[None, :]).argmin(axis=1)]

    alpha = 1.0
    tau_eps = 16.0
    tau_besag = 4.0
    tau_rw2 = 25.0

    y = alpha + u_obs + np.cos(z_binned) + rng.normal(0, 1.0 / np.sqrt(tau_eps), size=n_obs)

    effects = [
        {"model": "fixed", "n": 1, "theta_len": 0},
        {"model": "besag", "n": n_spatial, "theta_len": 1, "scale_model": True, "adj": adj},
        {
            "model": "rw2",
            "n": n_rw2,
            "theta_len": 1,
            "scale_model": True,
            "positions": z_bins.tolist(),
        },
    ]
    q_structured = core.build_structured_precision(
        effects, [float(np.log(tau_besag)), float(np.log(tau_rw2))], 1e-4
    )
    q_mat = q_structured.to_scipy().toarray()

    assert q_mat[0, 0] == 1e-4

    q_besag_raw = core.besag_precision_matrix(adj, 1.0)
    q_besag_scaled = core.scale_model_csc(q_besag_raw).to_scipy().toarray()
    np.testing.assert_allclose(
        q_mat[1 : 1 + n_spatial, 1 : 1 + n_spatial],
        tau_besag * q_besag_scaled,
        rtol=1e-12,
    )

    q_rw2_raw = core.crw2_precision_matrix(z_bins.tolist(), 1.0, layout="simple")
    q_rw2_scaled = core.scale_model_csc(q_rw2_raw).to_scipy().toarray()
    np.testing.assert_allclose(
        q_mat[1 + n_spatial :, 1 + n_spatial :],
        tau_rw2 * q_rw2_scaled + np.eye(n_rw2) * 1e-4,
        rtol=1e-10,
        atol=1e-12,
    )

    # R-INLA constr=TRUE: one sum-to-zero row per intrinsic field (Besag + RW2).
    constrs = core.structured_constraints(effects)
    assert constrs is not None
    ca, ce = constrs
    n_total = 1 + n_spatial + n_rw2
    assert len(ce) == 2
    assert len(ca) == 2 * n_total

    data = {
        "y": y,
        "region": region_id,
        "cov": z_binned,
        "adj_list": adj,
    }
    res = inla(
        "y ~ 1 + f(region, model='besag', scale_model=True) + f(cov, model='rw2', scale_model=True)",
        data=data,
        family="gaussian",
        deterministic=True,
    )

    est_besag = res.summary_random["region"]["mean"]
    est_rw2 = res.summary_random["cov"]["mean"]
    np.testing.assert_allclose(np.sum(est_besag), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.sum(est_rw2), 0.0, atol=1e-10)


def test_competing_besag_rw2_spatially_structured_covariate():
    """Competing Besag + RW2 where z is spatially structured (correlated with u)."""
    rng = np.random.default_rng(42)
    adj = _make_graph_19()
    n_spatial = 19
    n_obs = 200

    true_u = np.array(
        [
            -0.8,
            -0.6,
            -0.4,
            -0.3,
            -0.2,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.4,
            0.3,
            0.1,
            0.0,
            -0.2,
            -0.4,
            -0.6,
            -0.8,
        ]
    )
    true_u -= true_u.mean()

    z_base = true_u * 1.5 + rng.normal(0, 0.2, size=n_spatial)

    region_id = rng.integers(1, n_spatial + 1, size=n_obs)
    u_obs = np.array([true_u[r - 1] for r in region_id])
    z_obs = np.array([z_base[r - 1] for r in region_id]) + rng.normal(0, 0.05, size=n_obs)
    z_binned = inla.group(z_obs, n=20, method="cut")

    z_unique_bins = np.sort(np.unique(z_binned[np.isfinite(z_binned)]))
    true_f = 0.5 * (z_unique_bins**2) - 0.3
    true_f -= true_f.mean()
    f_map = {b: true_f[i] for i, b in enumerate(z_unique_bins)}
    f_obs = np.array([f_map[zb] for zb in z_binned])

    alpha = 1.5
    true_tau_obs = 16.0
    y = alpha + u_obs + f_obs + rng.normal(0, 1.0 / np.sqrt(true_tau_obs), size=n_obs)

    data = {
        "y": y,
        "region": region_id,
        "cov": z_binned,
        "adj_list": adj,
    }

    assert np.corrcoef(u_obs, z_binned)[0, 1] > 0.85

    res = inla(
        "y ~ 1 + f(region, model='besag', scale_model=True, hyper={'prec': {'prior': 'pc.prec', 'param': [1.0, 0.01]}}) "
        "+ f(cov, model='rw2', scale_model=True, hyper={'prec': {'prior': 'pc.prec', 'param': [1.0, 0.01]}})",
        data=data,
        family="gaussian",
        deterministic=True,
    )

    assert len(res.mode) == 3
    assert res.summary_hyperpar["names"][0] == "Precision for the Gaussian observations"
    assert res.summary_hyperpar["names"][1] == "Precision for region"
    assert res.summary_hyperpar["names"][2] == "Precision for cov"

    besag_est = np.asarray(res.summary_random["region"]["mean"], dtype=float)
    rw2_est = np.asarray(res.summary_random["cov"]["mean"], dtype=float)
    rw2_id = np.asarray(res.summary_random["cov"]["ID"], dtype=float)

    # Linear-in-z spatial confounding is identified with RW2 (R-INLA
    # constr=TRUE), so field-vs-simulator correlations are not the
    # right check — the reconstruction of u+f is.
    besag_obs = np.array([besag_est[int(r) - 1] for r in region_id])
    rw2_obs = np.interp(z_binned, rw2_id, rw2_est)
    recon = besag_obs + rw2_obs
    signal = u_obs + f_obs
    recon = recon - recon.mean() + signal.mean()
    corr_recon = float(np.corrcoef(recon, signal)[0, 1])
    assert corr_recon > 0.90, f"Besag+RW2 reconstruction corr {corr_recon} too low"

    np.testing.assert_allclose(res.summary_random["cov"]["ID"], z_unique_bins, atol=1e-12)
    assert len(res.summary_random["cov"]["ID"]) == len(z_unique_bins)


def test_rw2_absorbs_linear_spatial_confounder():
    """R-INLA leaves the RW2 linear trend free; a spatial linear-in-z signal
    must go into RW2, not Besag.
    """
    rng = np.random.default_rng(7)
    n_spatial = 12
    adj = [[(i - 1) % n_spatial, (i + 1) % n_spatial] for i in range(n_spatial)]
    n_obs = 120

    z_region = np.linspace(-1.5, 2.5, n_spatial)
    z_region -= z_region.mean()
    region_id = rng.integers(1, n_spatial + 1, size=n_obs)
    z_obs = np.array([z_region[r - 1] for r in region_id])
    y = 0.5 + 2.0 * z_obs + rng.normal(0.0, 0.15, size=n_obs)

    data = {
        "y": y,
        "region": region_id,
        "cov": z_obs,
        "adj_list": adj,
    }
    res = inla(
        "y ~ 1 + f(region, model='besag', scale_model=True) + f(cov, model='rw2', scale_model=True)",
        data=data,
        family="gaussian",
        deterministic=True,
    )

    besag = np.asarray(res.summary_random["region"]["mean"], dtype=float)
    rw2_id = np.asarray(res.summary_random["cov"]["ID"], dtype=float)
    rw2_mean = np.asarray(res.summary_random["cov"]["mean"], dtype=float)
    rw2_at_region = np.interp(z_region, rw2_id, rw2_mean)

    corr_rw2 = float(np.corrcoef(rw2_at_region, z_region)[0, 1])
    corr_besag = float(np.corrcoef(besag, z_region)[0, 1])
    assert corr_rw2 > 0.95, f"RW2 should take the linear spatial trend, corr={corr_rw2}"
    assert np.std(rw2_at_region) > 2.0 * np.std(besag), (
        f"Besag std={np.std(besag):.3f} should be small vs RW2 std={np.std(rw2_at_region):.3f}"
    )
    assert abs(corr_besag) < abs(corr_rw2)


def test_group_returns_bin_medians():
    x = np.array([1.0, 2.0, 3.0, 8.0, 9.0, 10.0])
    g = inla.group(x, n=2, method="cut")
    assert g.shape == x.shape
    # Two equal-length bins on [1, 10]: low cluster median 2, high cluster median 9.
    np.testing.assert_allclose(np.unique(g), np.array([2.0, 9.0]), atol=1e-12)


def test_classic_rinla_competing_besag_rw2(tmp_path):
    """Classic R-INLA vs py-inla on competing Besag+RW2 with a spatial covariate.

    Skips when Rscript or the INLA package is missing (not required in CI).
    """
    import csv
    import os
    import shutil
    import subprocess

    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if os.environ.get("INLA_SKIP_R_CONFORMANCE"):
        pytest.skip("INLA_SKIP_R_CONFORMANCE set")
    probe = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(requireNamespace("INLA", quietly=TRUE)))'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "TRUE":
        pytest.skip("classic INLA package not installed")

    rng = np.random.default_rng(2026)
    n = 12
    adj = [[(i - 1) % n, (i + 1) % n] for i in range(n)]
    n_obs = 96
    true_u = np.sin(np.arange(n) * 0.7)
    true_u -= true_u.mean()
    z_region = true_u + rng.normal(0, 0.15, size=n)
    region = rng.integers(1, n + 1, size=n_obs)
    z = np.array([z_region[r - 1] for r in region]) + rng.normal(0, 0.05, size=n_obs)
    z_g = inla.group(z, n=16, method="cut")
    y = (
        1.2
        + np.array([true_u[r - 1] for r in region])
        + 0.4 * (z_g**2)
        + rng.normal(0, 0.2, size=n_obs)
    )

    csv_path = tmp_path / "classic_besag_rw2.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "region", "z_g"])
        for row in zip(y, region, z_g, strict=True):
            w.writerow([repr(float(a)) for a in row])

    graph_path = tmp_path / "cycle.graph"
    lines = [str(n)]
    for i in range(n):
        nbs = [j + 1 for j in adj[i]]
        lines.append(f"{i + 1} {len(nbs)} " + " ".join(str(j) for j in nbs))
    graph_path.write_text("\n".join(lines) + "\n")

    r_script = tmp_path / "fit_classic.R"
    r_script.write_text(
        f"""
library(INLA)
df <- read.csv("{csv_path}")
g <- inla.read.graph("{graph_path}")
res <- inla(
  y ~ 1 +
    f(region, model = "besag", graph = g, scale.model = TRUE,
      hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01)))) +
    f(z_g, model = "rw2", scale.model = TRUE,
      hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01)))),
  data = df, family = "gaussian"
)
h <- res$summary.hyperpar$mean
cat(sprintf("hyper\\t%s\\n", paste(format(h, digits = 12), collapse = ",")))
cat(sprintf("besag\\t%s\\n", paste(format(res$summary.random$region$mean, digits = 12), collapse = ",")))
cat(sprintf("rw2_id\\t%s\\n", paste(format(res$summary.random$z_g$ID, digits = 12), collapse = ",")))
cat(sprintf("rw2\\t%s\\n", paste(format(res$summary.random$z_g$mean, digits = 12), collapse = ",")))
"""
    )
    proc = subprocess.run(["Rscript", str(r_script)], capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        pytest.fail(f"classic INLA fit failed:\n{proc.stdout}\n{proc.stderr}")

    parsed: dict[str, list[float]] = {}
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        key, payload = line.split("\t", 1)
        parsed[key] = [float(v) for v in payload.split(",") if v.strip()]

    py = inla(
        "y ~ 1 + f(region, model='besag', scale_model=True, hyper={'prec': {'prior': 'pc.prec', 'param': [1.0, 0.01]}}) "
        "+ f(z_g, model='rw2', scale_model=True, hyper={'prec': {'prior': 'pc.prec', 'param': [1.0, 0.01]}})",
        data={"y": y, "region": region, "z_g": z_g, "adj_list": adj},
        family="gaussian",
        deterministic=True,
    )
    py_hyper = np.asarray(py.summary_hyperpar["mean"], dtype=float)
    r_hyper = np.asarray(parsed["hyper"], dtype=float)
    np.testing.assert_allclose(py_hyper, r_hyper, rtol=0.15, atol=0.05)

    py_besag = np.asarray(py.summary_random["region"]["mean"], dtype=float)
    r_besag = np.asarray(parsed["besag"], dtype=float)
    corr = float(np.corrcoef(py_besag, r_besag)[0, 1])
    assert corr > 0.99, f"Besag vs classic R-INLA corr={corr}"

    py_id = np.asarray(py.summary_random["z_g"]["ID"], dtype=float)
    r_id = np.asarray(parsed["rw2_id"], dtype=float)
    np.testing.assert_allclose(np.sort(py_id), np.sort(r_id), rtol=1e-8, atol=1e-8)

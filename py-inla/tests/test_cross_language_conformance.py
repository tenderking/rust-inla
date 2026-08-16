"""R vs Python conformance: identical model, identical numbers.

Both front-ends are thin skins over the same Rust engine, so a fit specified the
same way must produce the same θ, marginal likelihood, information criteria and
summary tables. Divergence here means a control, default or summary field exists
on one side only — the class of bug that let WAIC drift between the bindings.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from inla.api import fit

REPO_ROOT = Path(__file__).resolve().parents[2]
R_DRIVER = Path(__file__).parent / "conformance" / "fit_models.R"
R_LIB = REPO_ROOT / "target" / "release" / "libinla_rs.so"

# Loose enough for platform math differences, tight enough to catch real drift.
RTOL = 1e-6
ATOL = 1e-8


def _dataset() -> dict[str, np.ndarray]:
    """Deterministic data shared by both front-ends (no RNG across languages)."""
    n = 24
    idx = np.arange(1, n + 1, dtype=float)
    y = np.sin(0.5 * idx) + 0.15 * np.cos(1.7 * idx) + 0.02 * idx
    count = np.abs(np.round(2.0 + 1.5 * np.sin(0.3 * idx))).astype(float)
    return {"y": y, "idx": idx, "count": count}


def _write_csv(data: dict[str, np.ndarray], path: Path) -> None:
    keys = list(data)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for row in zip(*(data[k] for k in keys)):
            writer.writerow([repr(float(v)) for v in row])


def _run_r(csv_path: Path) -> dict[tuple[str, str], list[float] | list[str]]:
    proc = subprocess.run(
        ["Rscript", str(R_DRIVER), str(REPO_ROOT), str(csv_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

    out: dict[tuple[str, str], list[float] | list[str]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        model, field, payload = parts
        values = [p.strip() for p in payload.split(",") if p.strip()]
        if field == "hyper_labels":
            out[(model, field)] = values
        else:
            out[(model, field)] = [float(v) for v in values]
    return out


def _python_fits(data: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "ar1": fit("y ~ -1 + f(idx, model='ar1', obs_precision=25.0)", data),
        "rw2": fit("y ~ -1 + f(idx, model='rw2', obs_precision=100.0)", data),
        "iid": fit("y ~ -1 + f(idx, model='iid', obs_precision=25.0)", data),
        "seasonal": fit("y ~ -1 + f(idx, model='seasonal', season=4, obs_precision=50.0)", data),
        "poisson_iid": fit(
            "count ~ -1 + f(idx, model='iid')",
            data,
            family="poisson",
            initial_theta=[1.0],
        ),
    }


@pytest.fixture(scope="module")
def conformance(tmp_path_factory):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not R_LIB.exists():
        pytest.skip(f"{R_LIB} not built (cargo build --release -p r-inla)")
    if os.environ.get("INLA_SKIP_R_CONFORMANCE"):
        pytest.skip("INLA_SKIP_R_CONFORMANCE set")

    data = _dataset()
    csv_path = tmp_path_factory.mktemp("conformance") / "data.csv"
    _write_csv(data, csv_path)
    return _run_r(csv_path), _python_fits(data)


MODELS = ["ar1", "rw2", "iid", "seasonal", "poisson_iid"]


@pytest.mark.parametrize("model", MODELS)
def test_theta_mode_matches(conformance, model):
    r_out, py_out = conformance
    np.testing.assert_allclose(
        np.asarray(py_out[model].mode, dtype=float),
        np.asarray(r_out[(model, "mode")], dtype=float),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{model}: hyperparameter mode differs between R and Python",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "field",
    ["mlik", "mlik_gaussian", "dic", "waic", "effective_params"],
)
def test_scalar_criteria_match(conformance, model, field):
    r_out, py_out = conformance
    attr = {
        "mlik": "marginal_log_lik",
        "mlik_gaussian": "marginal_log_lik_gaussian",
        "dic": "dic",
        "waic": "waic",
        "effective_params": "effective_params",
    }[field]
    py_value = float(getattr(py_out[model], attr))
    r_value = float(r_out[(model, field)][0])
    np.testing.assert_allclose(
        py_value,
        r_value,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{model}.{field}: R={r_value} Python={py_value}",
    )


@pytest.mark.parametrize("model", MODELS)
def test_hyperpar_summary_matches(conformance, model):
    r_out, py_out = conformance
    table = py_out[model].summary_hyperpar
    assert table is not None, f"{model}: Python has no summary_hyperpar"

    assert list(table["names"]) == list(r_out[(model, "hyper_labels")]), (
        f"{model}: hyperparameter labels differ; the registry should make these identical"
    )
    for field, key in (("mean", "hyper_mean"), ("sd", "hyper_sd")):
        np.testing.assert_allclose(
            np.asarray(table[field], dtype=float),
            np.asarray(r_out[(model, key)], dtype=float),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"{model}: natural-scale hyperparameter {field} differs",
        )


@pytest.mark.parametrize("model", MODELS)
def test_random_effect_means_match(conformance, model):
    r_out, py_out = conformance
    key = (model, "random_mean")
    assert key in r_out, f"{model}: R produced no summary.random"
    py_table = next(iter(py_out[model].summary_random.values()))
    np.testing.assert_allclose(
        np.asarray(py_table["mean"], dtype=float),
        np.asarray(r_out[key], dtype=float),
        rtol=1e-5,
        atol=1e-7,
        err_msg=f"{model}: random effect posterior means differ",
    )

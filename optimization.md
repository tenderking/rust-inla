# INLA / FGN Performance Optimizations

This note records the performance work done so that FGN inference at
`n ≈ 250` is practical in the R smoke test (roughly **40–60 s → ~1 s**
per fit in release).

## Context

Fractional Gaussian noise (FGN) has a **dense** Toeplitz covariance, so the
exact precision `Q = Σ⁻¹` is dense (`nnz = n²`). Exact FGN INLA is therefore
inherently **O(n³)** per hyperparameter evaluation. That is fine for moderate
`n` if the constants are sane; it is catastrophic if any step is accidentally
**O(n⁴)** or repeatedly rebuilds dense matrices through a sparse CSC API.

The R smoke path (`r-inla/smoke.sh`) exercises:

1. Build FGN precision `Q(θ)` for many `θ` (Nelder–Mead + Hessian + CCD).
2. For each `θ`, find the latent mode (Newton) and factor posterior precision.
3. At integration points, form latent marginal variances `diag(Q_post⁻¹)`.

---

## 1. Covariance inversion was O(n⁴)

**Where:** `inla_core::integration::invert_symmetric_matrix`, used by
`fgn_precision_csc`.

**Bug:** For each of the `n` columns of the identity, the code copied the full
matrix and re-ran Gaussian elimination. Cost ≈ `n × O(n³) = O(n⁴)`.

**Evidence (release, precision build only):**

| n   | Before (O(n⁴)) | After (O(n³)) |
|-----|----------------|---------------|
| 25  | ~0.7 ms        | ~0.09 ms      |
| 100 | ~50 ms         | ~1.4 ms       |
| 250 | **~1.1 s**     | **~12 ms**    |

**Fix:** Single Gauss–Jordan pass on the augmented system `[A | I]` → O(n³).

FGN additionally uses a dedicated **Cholesky** invert (`invert_spd_cholesky`)
because the covariance is SPD; that is stabler and cheaper than general GE.

---

## 2. Dense work was routed through CSC three times

**Where:** `laplace_newton_step`, `find_latent_mode`, `ldlt_factorize`.

**Problem:** Hot path looked like:

```
CSC prior → densify → build n² triplets → CSC → densify again → LDLT
```

For FGN the prior is already fully dense, so the sparse format only added
allocation and copying. A full O(n²) symmetry scan ran on every factorize as
well.

**Fix:**

- `csc_to_dense` / `ldlt_factorize_dense` — factorize row-major dense storage
  directly.
- `laplace_newton_step` densifies once, adds `−hess` on the diagonal, factorizes
  dense, solves in place.
- Symmetry check reduced to a cheap sample (not a full n×n pass).

---

## 3. Posterior factor was computed twice

**Where:** `find_latent_mode`.

**Problem:** Newton already factorizes the current posterior system
`(Q_prior − diag(hess))`. On convergence the code rebuilt that matrix as CSC
and factorized **again**.

For Gaussian observations the confirming Newton step is exactly that posterior
precision, so the second factorization was pure waste (and paid the CSC
roundtrip).

**Fix:** Return `(step, factor)` from `laplace_newton_step` and **reuse** the
factor when `‖step‖ < tol`. Prior log-determinant still needs one dense
factor of `Q_prior + εI`, without CSC rebuild.

---

## 4. Marginal variances used n independent solves

**Where:** Integration loop in `run_inla_inference`.

**Problem:**

```text
for i in 0..n:
    solve Q_post x = e_i
    var[i] = x[i]
```

That is O(n³) with poor constants and n allocations.

**Fix:** `ldlt_diagonal_inverse`:

1. Invert unit lower-triangular `L` once → `Y = L⁻¹` (O(n³)).
2. `(Q⁻¹)ᵢᵢ = Σ_{k≥i} Yₖᵢ² / Dₖ` (O(n²)).

Same asymptotics as n solves for dense `L`, but one structured pass and no
per-column allocations.

---

## 5. Smoke test ran a debug library

**Where:** `r-inla/smoke.sh`.

**Problem:** `cargo build` (dev) + `target/debug/librinla.so`. Debug FGN INLA
at `n = 250` was on the order of **~80 s**; release was already ~3 s even before
the dense-path cleanup.

**Fix:** Build and load release:

```bash
cargo build -p r-inla --release
# dyn.load ../target/release/librinla.so
```

---

## Combined result (smoke, n = 250, four Hurst values)

| Stage                         | Typical time          |
|------------------------------|------------------------|
| Before (debug + O(n⁴) + CSC) | ~40–60 s **per** fit   |
| After (release + fixes)      | **~0.8–1.3 s per fit** |

Point estimates in the smoke validation (H ∈ {0.6, 0.7, 0.8, 0.9}) are
unchanged by these optimizations; they only affect runtime and numerical path
equivalence for the dense algebra.

---

## What is *not* fixed (inherent / future work)

Exact FGN remains **dense**. Cost per hyperparameter eval stays Θ(n³) and
memory Θ(n²). That will not scale like AR(1)/RW (banded sparse).

**Sparse FGN approximation (partially done):** `fgn_approx_precision_csc`
interpolates the legacy `FGN_K3_PARAM` / `FGN_K4_PARAM` tables and builds a
sparse AR-mixture precision of size `(order+1)·n` (`order` ∈ {3,4}). The
matrix itself is sparse and matches R-INLA’s Qfunc_fgn path. However,
`ldlt_factorize` still **densifies** CSC → dense LDLT, so large-`n` FGN
approx fits do **not** yet get O(n) banded/sparse solve cost. Next step:
time-major reorder (bandwidth O(order)) or a true sparse LDLT/CHOLMOD.

Natural next steps if larger `n` is required:

1. **Sparse LDLT / banded factor** for FGN-approx (and AR1/RW/Besag) so CSC
   densification is never on that path.
2. **Toeplitz algorithms** for Σ (Levinson / Trench) if staying exact but
   wanting cheaper `Q` construction than generic Cholesky of a filled matrix.
3. **Gaussian closed form:** skip Newton confirmations and reuse algebra when
   the likelihood Hessian is constant.

---

## Files touched

| File | Change |
|------|--------|
| `crates/inla_core/src/integration.rs` | O(n³) `invert_symmetric_matrix` |
| `crates/inla_core/src/latent_models.rs` | FGN via Cholesky invert |
| `crates/inla_core/src/fgn.rs` / `fgn_tables.rs` | R-INLA AR-mixture FGN tables → sparse Q |
| `crates/inla_core/src/ldlt.rs` | Dense factor/solve, diag(Q⁻¹), Newton returns factor |
| `crates/inla_core/src/inference.rs` | Reuse Newton factor; fast variances |
| `crates/inla_core/src/lib.rs` | Re-exports |
| `r-inla/smoke.sh` | Release build + `librinla.so` path |

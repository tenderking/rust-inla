#!/usr/bin/env bash
# Create roadmap issues from deferred work formerly tracked in plan.md /
# optimization.md. Idempotent by title: skips titles that already exist.
#
# Requires: gh auth login
# Usage: ./scripts/seed-roadmap-issues.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install from https://cli.github.com/ then re-run." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Not logged in. Run: gh auth login" >&2
  exit 1
fi

existing="$(gh issue list --state all --limit 200 --json title -q '.[].title' 2>/dev/null || true)"

create_issue() {
  local title="$1"
  local body="$2"
  local labels="$3"
  if printf '%s\n' "$existing" | grep -Fxq "$title"; then
    echo "skip (exists): $title"
    return 0
  fi
  # shellcheck disable=SC2086
  gh issue create --title "$title" --body "$body" --label $labels
  echo "created: $title"
}

# Labels are created if missing (best-effort).
for label_spec in \
  "enhancement:New features and API surface" \
  "performance:Speed / memory work" \
  "r-binding:R / extendr front-end" \
  "python-binding:Python / PyO3 front-end" \
  "engine:Rust inference / math core"
do
  name="${label_spec%%:*}"
  desc="${label_spec#*:}"
  gh label create "$name" --description "$desc" --force >/dev/null 2>&1 || true
done

create_issue \
  "Support formula copy= / shared latent with free β" \
  "$(cat <<'EOF'
## Summary
R-INLA `copy` lets several formula terms share one latent field with a free scaling β. rust-inla does not implement this yet.

## Acceptance
- [ ] `copy=` (or equivalent) in R and Python formulas
- [ ] Shared latent layout in `ModelPlan` / structured Q
- [ ] Port + smoke covering β recovery

## Notes
Formerly tracked as partial/deferred in `plan.md`.
EOF
)" \
  "enhancement,engine"

create_issue \
  "R rgeneric: Q callbacks during hyperparameter optimisation" \
  "$(cat <<'EOF'
## Summary
Python `inla.define` can supply a custom Q during Nelder–Mead. R currently exposes `inla_rs_rgeneric_define()` but formula/callback optimisation during the outer loop is still thin.

## Acceptance
- [ ] R callback invoked for each θ evaluation in inference
- [ ] Smoke / conformance against a simple custom Q
- [ ] Document the R vs Python rgeneric surface

## Notes
Use Python for e2e custom Q until this lands.
EOF
)" \
  "enhancement,r-binding"

create_issue \
  "R multi-effect formula: f(model=\"spde\")" \
  "$(cat <<'EOF'
## Summary
Python supports `f(model='spde', ...)` in formulas. R still uses the dedicated `inla_rs_spde(...)` entry point and cannot mix SPDE into multi-effect structured formulas the same way.

## Acceptance
- [ ] `f(idx, model=\"spde\", ...)` in `inla_rs` formulas
- [ ] Mesh / projector wiring through the shared structured path
- [ ] Smoke covering SPDE + another latent or fixed effect
EOF
)" \
  "enhancement,r-binding"

create_issue \
  "R CRW2 layouts: pairs and block (not only simple)" \
  "$(cat <<'EOF'
## Summary
Python productizes CRW2 `layout` in {`simple`, `pairs`, `block`}. R structured inference still defaults to `simple`.

## Acceptance
- [ ] Expose `layout=` (or equivalent) on R CRW2 effects
- [ ] Match Python Q construction for `pairs` / `block`
- [ ] Smoke for non-simple layouts
EOF
)" \
  "enhancement,r-binding"

create_issue \
  "Sparse / banded LDLT path (stop densifying CSC on FGN approx and sparse GMRFs)" \
  "$(cat <<'EOF'
## Summary
`fgn_approx_precision_csc` builds a sparse AR-mixture Q, but factorisation still densifies CSC → dense LDLT. Large-`n` FGN approx and other sparse GMRFs (AR1/RW/Besag) should stay sparse.

## Acceptance
- [ ] Sparse (or banded) LDLT / CHOLMOD-style path for CSC priors
- [ ] Time-major reorder for FGN approx so bandwidth is O(order)
- [ ] Benchmark: FGN approx at n≈500+ without Θ(n³) dense factor

## Notes
Exact dense FGN remains Θ(n³) / Θ(n²) by construction; prefer `order=3/4` for large n.
Formerly future work in `optimization.md`.
EOF
)" \
  "performance,engine"

create_issue \
  "Toeplitz algorithms for exact FGN covariance / precision" \
  "$(cat <<'EOF'
## Summary
Exact FGN Σ is Toeplitz. Levinson / Trench (or similar) can build or apply Q cheaper than a generic dense Cholesky of a filled matrix when staying on the exact path.

## Acceptance
- [ ] Evaluate Levinson/Trench (or FFT Toeplitz) for Σ or Q construction
- [ ] Wire into exact FGN precision path when beneficial
- [ ] Document when exact vs `order=3/4` approx should be used
EOF
)" \
  "performance,engine"

create_issue \
  "Gaussian closed form: skip redundant Newton / reuse algebra when Hessian is constant" \
  "$(cat <<'EOF'
## Summary
For Gaussian observations the likelihood Hessian is constant. Newton confirmation and some re-factorisations can be skipped or algebra reused.

## Acceptance
- [ ] Detect constant-Hessian Gaussian case in the latent mode loop
- [ ] Avoid redundant posterior factors where safe
- [ ] No change to non-Gaussian families; smoke mlik/mode unchanged for Gaussian AR1/FGN
EOF
)" \
  "performance,engine"

create_issue \
  "Takahashi selective sparse inversion for latent marginal variances" \
  "$(cat <<'EOF'
## Summary
Full `diag(Q⁻¹)` via dense or full sparse inverse is more work than needed for INLA marginal variances. Takahashi (or faer’s selective inversion) can compute only the required entries.

## Acceptance
- [ ] Selective inversion for sparse posterior factors
- [ ] Use it in the CCD/grid integration variance path
- [ ] Benchmark vs current dense/sparse diag path on AR1 / Besag
EOF
)" \
  "performance,engine"

echo "Done. Review issues at: $(gh repo view --json url -q .url)/issues"

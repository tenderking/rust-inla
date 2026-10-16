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

echo "Done. Review issues at: $(gh repo view --json url -q .url)/issues"

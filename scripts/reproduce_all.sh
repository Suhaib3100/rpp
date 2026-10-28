#!/usr/bin/env bash
# Reproduce the entire SAGE-HE project end to end.
#
# Safe to re-run: every stage skips work that is already complete (experiments
# with a result.json are not retrained), so an interrupted run resumes.
#
#   bash scripts/setup_env.sh      # once, creates .venv
#   bash scripts/reproduce_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
mkdir -p logs

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

# Experiment suites are run with `|| warn`: a single failed run should not
# abort a multi-hour reproduction. Failures are collected and reported at the
# end, and every suite is resumable, so re-running picks up where it stopped.
FAILURES=""
warn() { FAILURES="$FAILURES\n  - $1"; printf '\033[1;33m!! %s failed (continuing)\033[0m\n' "$1"; }

step "1/10  Acquire and verify the dataset"
$PY scripts/download_dataset.py
$PY scripts/make_splits.py

step "2/10  Dataset integrity and leakage audit"
$PY scripts/verify_dataset.py

step "3/10  Verify every citation against Crossref / arXiv"
$PY scripts/verify_citations.py

step "4/10  Main experiments (baseline + SAGE-HE, 3 seeds each)"
$PY scripts/run_suite.py main 2>&1 | tee -a logs/main.log || warn "suite:main"

step "5/10  Learning curve, ablation and design studies"
$PY scripts/run_suite.py capacity 2>&1 | tee -a logs/capacity.log || warn "suite:capacity"
$PY scripts/run_suite.py ablation 2>&1 | tee -a logs/ablation.log || warn "suite:ablation"
$PY scripts/run_suite.py design   2>&1 | tee -a logs/design.log || warn "suite:design"
$PY scripts/run_suite.py component 2>&1 | tee -a logs/component.log || warn "suite:component"
$PY scripts/run_suite.py sensitivity 2>&1 | tee -a logs/sensitivity.log || warn "suite:sensitivity"

step "6/10  Hyperparameter optimisation"
$PY scripts/run_optuna.py 2>&1 | tee -a logs/optuna.log || warn "optuna"

step "7/10  Encrypted inference benchmark"
$PY scripts/run_encrypted_eval.py --sweep 2>&1 | tee -a logs/encryption.log || warn "encrypted-eval"

step "8/10  Explainability"
$PY scripts/make_gradcam.py 2>&1 | tee -a logs/gradcam.log || warn "gradcam"

step "9/10  Figures, diagrams and report"
$PY scripts/verify_diagram_shapes.py
$PY scripts/analyse_components.py
$PY scripts/make_figures.py
$PY scripts/make_comparisons.py
$PY scripts/make_diagrams.py
$PY scripts/make_report.py
$PY scripts/make_deliverables.py
$PY scripts/export_checkpoints.py

step "10/10 Presentation and visual QA"
$PY scripts/make_presentation.py
$PY scripts/render_slides.py || echo "layout checks reported problems (see above)"

step "Tests"
$PY -m pytest -q

if [ -n "$FAILURES" ]; then
  printf '\n\033[1;33mFinished with failures.\033[0m\n'
else
  printf '\n\033[1;32mAll stages complete.\033[0m\n'
fi
printf 'Deliverables:\n'
printf '  presentation/SAGE-HE.pptx        the deck\n'
printf '  presentation/contact_sheet.png   every slide, rendered\n'
printf '  reports/technical_report.md      generated from results\n'
printf '  reports/literature_review.md     novelty assessment\n'
printf '  results/                         metrics, figures, benchmarks\n'
printf '  reports/deliverables.md          achievements, limitations, blockers\n'
printf '  diagrams/                        12 architecture diagrams (SVG + PNG)\n'

if [ -n "$FAILURES" ]; then
  printf '\n\033[1;33mStages that failed (re-run this script to retry; completed work is skipped):\033[0m'
  printf "$FAILURES\n"
fi

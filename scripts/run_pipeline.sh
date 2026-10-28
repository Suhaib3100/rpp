#!/usr/bin/env bash
# Sequential driver for the whole experiment pipeline.
#
# Strictly one stage at a time. An earlier version polled for the previous
# stage's process and raced: it started a second training run while the first
# was still finishing, and on an 8 GB box with no swap the pair was enough to
# lose a run. flock makes overlap impossible rather than unlikely.
#
# Safe to re-run: stages with a result.json are skipped, and interrupted
# training resumes from last.pt.
set -u
export OMP_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=/home/vercel-sandbox/rpp
cd /home/vercel-sandbox/rpp
PY=.venv/bin/python
mkdir -p logs

exec 9>logs/.pipeline.lock
if ! flock -n 9; then
  echo "another pipeline run holds the lock; exiting"
  exit 1
fi

say() { echo "[$(date +%H:%M:%S)] $*"; }

run() {               # run <label> <logfile> <command...>
  local label="$1" log="$2"; shift 2
  say "=== $label ==="
  "$@" >> "$log" 2>&1
  local rc=$?
  say "$label finished (exit $rc)"
  return $rc
}

run "main suite"       logs/main.log      $PY scripts/run_suite.py main
run "capacity suite"   logs/capacity.log  $PY scripts/run_suite.py capacity
run "ablation suite"   logs/ablation.log  $PY scripts/run_suite.py ablation
run "design suite"     logs/design.log    $PY scripts/run_suite.py design
run "optuna study"     logs/optuna.log    $PY scripts/run_optuna.py --trials 15 --fraction 0.10 --epochs 8
run "encrypted eval"   logs/encryption.log $PY scripts/run_encrypted_eval.py --sweep
run "explainability"   logs/gradcam.log   $PY scripts/make_gradcam.py

say "=== build deliverables ==="
$PY scripts/verify_diagram_shapes.py >> logs/build.log 2>&1
$PY scripts/make_figures.py          >> logs/build.log 2>&1
$PY scripts/make_comparisons.py          >> logs/build.log 2>&1
$PY scripts/make_diagrams.py         >> logs/build.log 2>&1
$PY scripts/make_report.py           >> logs/build.log 2>&1
$PY scripts/make_presentation.py     >> logs/build.log 2>&1
$PY scripts/render_slides.py         >> logs/build.log 2>&1
say "=== PIPELINE COMPLETE ==="

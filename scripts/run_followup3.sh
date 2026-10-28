#!/usr/bin/env bash
# Third follow-up: re-run the sensitivity config that crashed, then rebuild.
#
# sensitivity/jia_ch128 failed with "conv_channels=128 cannot be pooled to
# fc_in=256" — a real bug in JiaChunkCNN's pooling arithmetic, now fixed and
# covered by a regression test. The fix preserves the pool shapes of every
# already-reported run (16->(4,4), 32->(2,4), 64->(1,4)), so no existing result
# is invalidated; only the missing run needs to be produced.
#
# Queues on the same lock so it can never overlap the suite still in flight.
set -u
export OMP_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=/home/vercel-sandbox/rpp
cd /home/vercel-sandbox/rpp
PY=.venv/bin/python
mkdir -p logs

exec 9>logs/.pipeline.lock
flock 9

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== re-run sensitivity (completed configs are skipped) ==="
$PY scripts/run_suite.py sensitivity >> logs/sensitivity.log 2>&1
say "sensitivity finished (exit $?)"

say "=== definitive rebuild ==="
$PY scripts/analyse_components.py    >> logs/build.log 2>&1
$PY scripts/verify_diagram_shapes.py >> logs/build.log 2>&1
$PY scripts/make_figures.py          >> logs/build.log 2>&1
$PY scripts/make_comparisons.py      >> logs/build.log 2>&1
$PY scripts/make_diagrams.py         >> logs/build.log 2>&1
$PY scripts/make_report.py           >> logs/build.log 2>&1
$PY scripts/make_deliverables.py     >> logs/build.log 2>&1
$PY scripts/make_presentation.py     >> logs/build.log 2>&1
$PY scripts/render_slides.py         >> logs/build.log 2>&1
$PY -m pytest -q                     >> logs/build.log 2>&1
say "=== FINAL BUILD COMPLETE ==="

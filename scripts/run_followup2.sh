#!/usr/bin/env bash
# Second follow-up: baseline sensitivity, then the definitive rebuild.
#
# Tests whether the failed reproduction of the published 97.1% is an artefact of
# the assumptions the paper forced on us (convolution width, activation) before
# that claim is allowed to stand. Queues behind the earlier stages on the same
# lock so nothing ever runs concurrently.
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

say "=== baseline sensitivity ==="
$PY scripts/run_suite.py sensitivity >> logs/sensitivity.log 2>&1
say "sensitivity finished (exit $?)"

say "=== final rebuild ==="
$PY scripts/analyse_components.py    >> logs/build.log 2>&1
$PY scripts/verify_diagram_shapes.py >> logs/build.log 2>&1
$PY scripts/make_figures.py          >> logs/build.log 2>&1
$PY scripts/make_comparisons.py      >> logs/build.log 2>&1
$PY scripts/make_diagrams.py         >> logs/build.log 2>&1
$PY scripts/make_report.py           >> logs/build.log 2>&1
$PY scripts/make_deliverables.py     >> logs/build.log 2>&1
$PY scripts/make_presentation.py     >> logs/build.log 2>&1
$PY scripts/render_slides.py         >> logs/build.log 2>&1
say "=== EVERYTHING COMPLETE ==="

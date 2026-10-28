#!/usr/bin/env bash
# Follow-up stage: the targeted component study, the component analysis, then a
# full rebuild of every derived artefact.
#
# Waits on the main pipeline's lock so the two can never run concurrently. The
# component study answers the question the 5% ablation raised — does the
# proposed architecture beat its own backbone at any measurable data size?
set -u
export OMP_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=/home/vercel-sandbox/rpp
cd /home/vercel-sandbox/rpp
PY=.venv/bin/python
mkdir -p logs

exec 9>logs/.pipeline.lock
flock 9                     # block until the main pipeline releases it

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== component study ==="
$PY scripts/run_suite.py component >> logs/component.log 2>&1
say "component study finished (exit $?)"

say "=== component analysis ==="
$PY scripts/analyse_components.py >> logs/component.log 2>&1
say "component analysis finished (exit $?)"

say "=== rebuild all derived artefacts ==="
$PY scripts/verify_diagram_shapes.py >> logs/build.log 2>&1
$PY scripts/make_figures.py          >> logs/build.log 2>&1
$PY scripts/make_comparisons.py      >> logs/build.log 2>&1
$PY scripts/make_diagrams.py         >> logs/build.log 2>&1
$PY scripts/make_report.py           >> logs/build.log 2>&1
$PY scripts/make_presentation.py     >> logs/build.log 2>&1
$PY scripts/render_slides.py         >> logs/build.log 2>&1
say "=== FOLLOW-UP COMPLETE ==="

# Configurations

Experiments in this project are defined as **`ExperimentConfig` dataclasses**
(`src/training/experiment.py`) and grouped into suites in
`scripts/run_suite.py`, rather than as free-floating YAML. The reason is that a
dataclass is type-checked at construction and shows up in `result.json`
verbatim, so every run carries the exact configuration that produced it — there
is no way for a config file and a result to drift apart.

The YAML files here are the **human-readable record** of the four suites, kept
in sync with `scripts/run_suite.py`. They document what was run; they are not
read at runtime.

| File | Suite | What it covers |
|---|---|---|
| `main.yaml` | `run_suite.py main` | Baseline reproduction + SAGE-HE, 3 seeds each, full data |
| `capacity.yaml` | `run_suite.py capacity` | Learning curve across training-set fractions |
| `ablation.yaml` | `run_suite.py ablation` | The nine-stage component ablation |
| `design.yaml` | `run_suite.py design` | Resolution, contrast, compression and top-k studies |
| `component.yaml` | `run_suite.py component` | Backbone alone vs the full model, matched seeds |

Every field defaults from `ExperimentConfig`; only overrides are listed.

```bash
.venv/bin/python scripts/run_suite.py main
.venv/bin/python scripts/run_suite.py ablation --fraction 0.15
.venv/bin/python scripts/run_suite.py design --only compress
```

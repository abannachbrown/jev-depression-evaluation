# jev-depression-evaluation

Reproducible evaluation of AutoTrust JEV-9B and JEV-27B on JSON states,
using hosted inference initially and a separate local backend later.

## Setup

Requires Git, uv, and Python 3.13.

```sh
uv sync --locked
cp .env.example .env
uv run python -c "import jev_eval"
```

Fill in `.env` locally. Never commit credentials.

## Configuration

- `configs/evaluations/`: datasets, model selection, and evaluation settings.
- `configs/backends/`: hosted and local inference settings.
- `configs/models/`: model identities and backend-specific references.

Paths in YAML resolve from the repository root. Fields ending in `_env`
name environment variables for the configuration loader to read.
An unset or empty optional directory override uses the configured path.
YAML itself does not expand environment variables.

The synthetic profile uses two invented states. The depression profile defaults
to individual JSON files under `data/private/states`. Each state uses the draft
`study_id`, `title`, and `abstract` fields. Confirm the final dataset schema and
exact count before a full evaluation.

## Project status

The six research notebooks use reusable preparation, configuration, inference,
evaluation and reporting functions under `src/jev_eval/`. For colleague review,
start with [`05_study_summary.ipynb`](notebooks/05_study_summary.ipynb): it reads
the archived full runs and summarises interim exploratory results without API
calls. Validation on 2,000 unseen studies is planned after colleague review.
For the implementation workflow, start with [`00_setup.ipynb`](notebooks/00_setup.ipynb)
and follow notebooks 01–04. See the [notebook guide](notebooks/README.md) for
inputs, permissions and troubleshooting.

```sh
uv sync --locked --group notebooks
uv run --locked --group notebooks jupyter lab
```

Select the kernel from this repository's `.venv`. Notebooks 04–05 read saved real
results without API calls and default to no exports. Notebooks 02–03 have
separate live settings and enabled execution flags; their model cells can incur
charges. The synthetic teaching fixtures remain available separately.

The supplied binary and hybrid NOUL request templates are preserved under
[`prompts/drafts/`](prompts/drafts/README.md). The binary template is valid JSON;
the original hybrid template is truncated. A separately supplied
`hybrid_noul_corrected.json` is complete JSON and preserved unchanged, but is not
selected by the binary workflow. The original synthetic citation state is under
`data/examples/`.

The [Inference.sh integration](docs/inference_sh.md) supports one synthetic state
and question per request to `autotrust/jev-9b` or `autotrust/jev-27b-v1-1`. Use
[`02_model_connection.ipynb`](notebooks/02_model_connection.ipynb) and
`configs/connections/inference_sh.yaml`; select a model with `model_config_file`.
Mock transport is available without credentials or charges. Live calls need
`INFERENCE_API_KEY` and explicit permission in both connection YAML and the cell.
Probabilities and raw response bytes are retained; optional exports stay private.

Notebook 03 supports separate ten-study pilots and full-cohort Inference.sh runs,
with checkpoints, raw-response preservation and explicit execution permissions.
Other generic hosted providers still need an adapter; local inference remains unimplemented.
Binary scoring is executable; hybrid scoring remains undefined.

## Reference labels and evaluation

Full-cohort analysis uses the prepared CSV, joined by original `study_id`, with
human inclusion status mapped explicitly to retain/exclude. The synthetic
teaching fixture uses reference JSONL. The positive class means retain for
full-text assessment. See [`data/README.md`](data/README.md).

Notebook 04 reports missing, failed, and invalid predictions alongside sensitivity,
specificity, precision, F1, balanced accuracy, and confusion counts. Headline
metrics are withheld by default when labelled prediction coverage is incomplete.

## Development

```sh
uv run ruff check src
uv run ruff format --check src
uv run --locked --group notebooks pytest
```

The tests include fresh-kernel offline execution of every notebook.
Provider tests use mocked responses only; no real API calls are made by tests.

## Commit protections

```sh
uv run --locked pre-commit install
uv run --locked python scripts/check_repository.py --working
```

The installed hook checks staged content and rejects saved notebook outputs,
extra execution metadata, protected paths, and likely credentials. Clear all
notebook outputs before saving and staging. Each clone needs the hook installed;
GitHub Actions repeats checks against committed content. Do not paste private
data into notebook source. These controls cannot identify every sensitive passage
or prevent deliberate hook bypasses.

## Data and outputs

Only synthetic example data belongs in Git. Keep real datasets under
`data/private/` or outside the repository. Results belong under `results/`.

Local inference is disabled by default. Model weights and GPU inference
packages are not included. Keep hosted and local adapters in their respective
packages under `src/jev_eval/backends/`.

Generated tables, figures, run records, and executed notebook copies belong under
`results/`. No model training or fine-tuning workflow is included.

See [`docs/reproducibility.md`](docs/reproducibility.md) for run-record
requirements and [`data/README.md`](data/README.md) for data storage rules.

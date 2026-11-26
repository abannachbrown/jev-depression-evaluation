# JEV depression citation-screening evaluation

This project evaluates whether AutoTrust JEV models can help identify citations
to retain for full-text assessment in a systematic review of animal experiments
related to depression. It compares model responses with human screening decisions
and records the inputs, settings and outputs needed to explain the evaluation.

The numbered Jupyter Notebooks
contain explanations and executable steps. Reusable preparation and calculations
live in the Python package under `src/jev_eval/`.

**Retain means “keep for full-text assessment”, including uncertain cases.** It
does not mean a study is finally eligible. An incorrect exclusion during
title-and-abstract screening can remove a study before a reviewer sees its full
text. This project therefore examines recall alongside screening workload.

The current work is exploratory. Colleague review and protocol clarification
precede the planned evaluation on **2,000 unseen studies**.

## Choose your route

| Your aim | Where to start | What you need |
| --- | --- | --- |
| Read an existing report | Open a privately supplied HTML/PDF report | No Python or API key |
| Reproduce saved-result tables | Install below; run notebook 00, then 04 or 05 | Prepared data, reference CSV and saved inference artifacts |
| Learn using invented data | Run 00; select `DATA_MODE = "demo"` in 01 and inspect the four mock examples in 02 | Repository and packages; no API key |
| Collect new model responses | Preparation, connection check, pilot, then full inference | Approved data, reviewed questions, API key and paid-call permissions |

**Notebook 03 now shows all four arms together.** Its four execution flags are
`False`. Existing binary and hybrid archives are reused even if an execution
flag is enabled; no successful paid call is repeated to recreate a comparison.
Some underlying workflow YAMLs still permit paid calls, so do not delete run
directories or change run IDs to trigger execution. Notebooks 04 and 05 analyse
saved outputs without making API calls.

## 1. Install the software once

### What the tools mean

| Tool or file | Purpose |
| --- | --- |
| Python | The language performing the preparation and calculations |
| Package | An installed collection of functions, such as pandas for tables |
| uv | Installs the project environment and runs commands within it |
| `.venv/` | This project's isolated Python environment, created locally |
| `pyproject.toml` | Lists the project and required packages |
| `uv.lock` | Records resolved package versions for reproducible installation |
| Git / GitHub | Record code changes / host the private repository |
| JupyterLab | Opens notebooks containing explanations, code and outputs |
| Kernel | The Python process executing a notebook's code |

You need private-repository access and an internet connection for installation.
Hosted inference runs remotely: no GPU, model weights or local GPU packages are
required. Local inference is a future extension; training and fine-tuning are
outside this workflow.

### Install Git and uv

Open **Terminal** on macOS or **PowerShell** on Windows. Terminal commands below
belong there, not in notebook cells. Run one line at a time; do not add a `$`
prompt. First check whether the tools are already installed:

```sh
git --version
uv --version
```

If needed, follow the [official Git installation guide](https://git-scm.com/install/)
and [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
The uv standalone installer for macOS/Linux is:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reopen the terminal afterwards and repeat the version checks. Follow your
institution's software installation arrangements where applicable.

### Download the repository, or use your existing copy

If you do **not** already have the project, navigate to a folder for your
research projects and run:

```sh
git clone https://github.com/abannachbrown/jev-depression-evaluation.git
cd jev-depression-evaluation
```

GitHub may ask you to authenticate. “Repository not found” can mean the signed-in
account lacks access to this private repository.

If you already have the project, open a terminal in that existing folder instead
of cloning again. In VS Code, open the project folder, then choose
**Terminal → New Terminal**. The folder containing `README.md`, `pyproject.toml`,
`uv.lock`, `src/` and `notebooks/` is the **repository root**. Run subsequent
terminal commands from there.

### Install Python and all required packages

The project uses Python **3.13**, recorded in `.python-version`. uv can install
that interpreter and create the environment:

```sh
uv python install 3.13
uv sync --locked --group notebooks
uv run --locked --group notebooks python -c "import jev_eval; print('JEV evaluation package is ready')"
```

The first installation can take several minutes. `uv sync` installs the project
into `.venv/`; `--group notebooks` includes the notebook packages. `--locked`
requires the existing lockfile to match the project configuration rather than
silently updating it. Keep `pyproject.toml` and `uv.lock` together. See
[uv's project guide](https://docs.astral.sh/uv/guides/projects/) for environment
and lockfile behaviour.

The command installs the packages together; do not install them individually:

| Packages | Use in this project |
| --- | --- |
| pandas, scikit-learn | Tables and evaluation metrics |
| httpx | Hosted API communication |
| Pydantic, PyYAML, python-dotenv | Validated settings and local environment variables |
| JupyterLab, ipykernel, nbformat, nbclient | Opening and executing notebooks |
| matplotlib, openpyxl | Figures and reading Excel workbooks |
| pytest, Ruff, pre-commit | Automated checks and Git commit protections |

Avoid adding `pip install` cells or independently upgrading packages while
reproducing a run. Package changes should be reviewed and recorded in the project
configuration and lockfile. Do not synchronise or rebuild the environment while
a live inference kernel is running.

## 2. Open and run a notebook

From the repository root:

```sh
uv run --locked --group notebooks jupyter lab
```

Keep the terminal open. JupyterLab opens in your browser; if it does not, open
the local URL printed in the terminal. Double-click `notebooks/`, then
`00_setup.ipynb`. This follows
[uv's project-based Jupyter workflow](https://docs.astral.sh/uv/guides/integration/jupyter/).

A **Markdown cell** explains the method. A **code cell** executes Python and
shows its output below. Read the explanation, select the next code cell and
press **Shift + Enter**. Work from top to bottom. A busy indicator or `[*]`
means the cell is running; do not repeatedly click Run. Later cells can depend
on variables created earlier. The
[JupyterLab notebook guide](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html)
explains the interface.

Choose the kernel belonging to this project's `.venv`. In VS Code, install its
Python and Jupyter extensions, open an `.ipynb` file and click **Select Kernel**.
Select `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.
Notebook 00 checks the environment. To inspect the interpreter yourself, run
this in a **notebook code cell**:

```python
import sys
print(sys.executable)
```

The path should be inside this project's `.venv`. Restart the kernel after
changing code, packages, settings or credentials, then rerun prerequisites.
Restarting clears variables, not saved result files. For long runs, keep the
computer awake and connected. Closing a notebook tab does not necessarily stop
its kernel; see [JupyterLab's kernel guide](https://jupyterlab.readthedocs.io/en/stable/user/running.html).

## 3. Obtain private inputs and configure credentials

### Data and results are supplied separately

GitHub contains code, configuration, reviewed questions and invented examples.
It does **not** contain the real workbook, prepared cohort, human decisions or
generated model results. Obtain these through the agreed private transfer route.
A fresh clone cannot reproduce real performance using synthetic examples alone.

The prepared research bundle contains:

| File | Contents | Methodological purpose |
| --- | --- | --- |
| `states.jsonl` | One study per line: original `study_id`, title and abstract | Model inputs, without human decisions |
| `reference_labels.csv` | Original study ID, human inclusion status and ITEM_ID | Separate human comparison reference |
| `manifest.json` | Source checksum, counts and validation/conversion information | Identifies which source and conversion produced the data |

JSONL means one JSON object per line. CSV is a table that Excel can also open.
Join results to human decisions by **`study_id`**, preserving its original text
value; ITEM_ID is not the join key. Keeping reference decisions out of the model
input prevents the model from being shown the answer against which it is tested.

The current data paths are in
[`configs/evaluations/full_hybrid_v1_jev_9b.yaml`](configs/evaluations/full_hybrid_v1_jev_9b.yaml)
and its JEV-27B equivalent. All four arms must use the same prepared bundle. Place
it under the configured `data/private/prepared/` location. To analyse saved
runs, also obtain the corresponding `results/full/` artifacts: cohort/model
manifests, predictions, response records and raw receipts. Do not overwrite an
existing run with a different supplied copy.

The Excel conversion reads the source without changing it and reports its
discrepancy worksheet without automatically adjudicating decisions. Its
configuration is
[`configs/data_preparation/depression_excel.yaml`](configs/data_preparation/depression_excel.yaml).
Notebook 01 currently **validates the prepared conversion**; it does not
recreate it automatically from Excel. For a new workbook, first review its path,
checksum, worksheets, column mapping and validation settings, then use the
conversion functions in [`src/jev_eval/data.py`](src/jev_eval/data.py).

### Add an API key only for live model calls

Notebooks 04 and 05 read files without contacting the API and need no API key.
For live calls, copy `.env.example` to `.env` **only if `.env` does not already
exist**. On macOS/Linux:

```sh
cp .env.example .env
```

On Windows, copy it in your editor or File Explorer and name the copy `.env`.
Open it locally and fill in `INFERENCE_API_KEY` with your Inference.sh key.
The generic `JEV_API_KEY` variable is not used by this backend. Never paste a
key into notebook code, outputs, GitHub or reports. Restart the kernel after
editing `.env`; an existing environment variable takes precedence over the file.

## 4. Follow the research workflow

| Notebook | What you do | Why it matters for systematic-review screening |
| --- | --- | --- |
| [00_setup](notebooks/00_setup.ipynb) | Check one environment and display all four experiment configurations | Makes shared and arm-specific settings explicit before data loading |
| [01_prepare_data](notebooks/01_prepare_data.ipynb) | Verify identical ordered citations and human references across all four arms | Avoids preprocessing, linkage or reference-mapping differences confounding comparisons |
| [02_model_connection](notebooks/02_model_connection.ipynb) | Inspect four mock requests/responses and existing saved connection metadata | Explains both designs and models without repeating paid connection tests |
| [03_run_models](notebooks/03_run_models.ipynb) | Review/reuse all four full archives, with separate optional execution cells | Makes the two-by-two design explicit without repeating paid calls |
| [04_evaluation](notebooks/04_evaluation.ipynb) | Compare saved scores with human decisions | Measures missed reference-retain studies and screening workload |
| [05_study_summary](notebooks/05_study_summary.ipynb) | Assemble an interim report | Makes methods, exceptions and unresolved decisions visible before validation |

Each notebook reloads its prerequisites; you do not need all six kernels open.
For colleague review, run 00 then 05 with the private artifacts available.
Missing research inputs cause prerequisite errors rather than silently being
replaced with invented data.

### Model choice and question design are different decisions

The hosted deployments are **JEV-9B** (`autotrust/jev-9b`) and **JEV-27B v1.1**
(`autotrust/jev-27b-v1-1`). Either can answer the selected screening questions.
A **NOUL** here is a natural-language yes/no question returning a probability
of “true”. Binary and hybrid describe the question design, not different models:

- **Binary NOUL:** one overall retain/exclude question.
- **Hybrid NOUL:** seven questions covering primary research, animal study,
  in vivo experiment, depression model, control group, eligible outcomes and
  overall eligibility.

The [reviewed v1 templates](prompts/reviewed/v1/README.md) have identical overall
eligibility definitions for binary and hybrid. The four-arm comparison explicitly reuses older binary runs with different
wording. Each retains its original identity: differences from hybrid v1 cannot
be attributed only to question design.

Hybrid analysis currently examines:

- **Direct:** the `overall_eligibility` answer.
- **R2:** the minimum of the first four primary-domain probabilities.
- **R3:** the mean of those four probabilities.

Control-group and outcome questions are secondary and do not enter R2/R3.
Derived scores are not established calibrated joint probabilities. R1 remains
disabled pending protocol clarification. Apply the reviewed instructions to
ambiguous studies; missing text or a technical failure is not an eligibility
exclusion.

### Find the setting that controls your step

YAML files are readable settings files. Preserve their indentation. Python code
uses `True`/`False`; YAML uses `true`/`false`. Paths resolve from the repository
root. Fields ending in `_env` name environment variables; YAML does not expand
shell variables itself.

| Setting | File or folder |
| --- | --- |
| Default setup approach, pilot/full profiles and R1 status | [`configs/screening/v1.yaml`](configs/screening/v1.yaml) |
| Both archive sources, four-arm recall target and comparison export | [`configs/evaluations/comparison.yaml`](configs/evaluations/comparison.yaml) |
| Generic synthetic demonstration | [`configs/notebooks.yaml`](configs/notebooks.yaml) |
| Single-question / hybrid connection | `configs/connections/inference_sh.yaml` / `hybrid_v1.yaml` |
| Four synthetic mock examples | [`configs/connections/comparison.yaml`](configs/connections/comparison.yaml), using the separate `comparison_binary.yaml` mock profile |
| Model identities | `configs/models/jev-9b.yaml` and `jev-27b.yaml` |
| Dataset, task and model selection | `configs/evaluations/` |
| Paid-call permission, budgets, saving, run ID and full dry-run policy | Selected file under `configs/workflows/` |
| Full hybrid analysis threshold | [`configs/evaluations/analysis_hybrid_v1.yaml`](configs/evaluations/analysis_hybrid_v1.yaml) |
| Interim report and export settings | [`configs/evaluations/interim_summary.yaml`](configs/evaluations/interim_summary.yaml) |

There is **no single global demo/live switch**. `configs/notebooks.yaml` controls
the generic demonstration. Notebooks 03–05 use explicit binary and hybrid
archive sources in the comparison YAML, independently of `default_approach`.
Connection mode is `mock`/`live`; batch workflow mode is `demo`/`hosted`/`local`.
Check the printed loaded profile, not an old saved output.

### Synthetic connection checks

Notebook 02 uses invented studies, not evaluation data. Both paid-call flags are
false. Its original one-question walkthrough now selects a dedicated mock binary
profile by default; the original live YAML is retained separately. The hybrid
profile remains mock by default. Section 9 shows four explicitly labelled
binary/hybrid × model examples with one shared synthetic state and rejects
live/paid/write-enabled profiles. Fabricated probabilities do not measure model
performance. Section 10 inventories saved connection metadata without resending
requests or altering previous responses.

Notebook 00's four-arm table is configuration-only and works before private data
are supplied. Notebook 01 defaults to `DATA_MODE = "prepared"` and validates the
actual cohort/reference equivalence across all four arms without converting the
workbook again. Its deterministic sample is a read-only preview, not a new
frozen pilot or replacement tuning split. Demo mode explicitly leaves actual
four-arm data equivalence unverified.

A live connection test needs the key, the connection YAML paid/write permissions
and its notebook flag. It makes one potentially billable request. See the
[notebook guide](notebooks/README.md) for the specific settings.

### Four-arm review and optional execution in notebook 03

The comparison has four primary arms: binary JEV-9B, binary JEV-27B v1.1,
hybrid JEV-9B and hybrid JEV-27B v1.1. It reuses the original binary archives
and the existing hybrid-v1 archives. No new matched-binary run is assumed.

1. Restart the project kernel after source changes.
2. Run **Sections 1–3** to load the four sources, review criteria and inspect
   coverage, exceptions and execution metadata. These sections make no API calls.
3. **Section 4** has a separate cell for each arm. All four `RUN_BINARY_*` and
   `RUN_HYBRID_*` flags are false. An existing archive returns `reuse_saved_run`
   without calling the provider, even if its flag is enabled.
4. **Section 5** reloads saved coverage. Proceed to 04 for the comparison and
   05 for the interim report.

For an entirely absent arm, changing only its explicit flag to true can make
billable calls and write results if its original YAML permissions and dry-run
policy also permit execution. Binary sends one question per study; hybrid sends
seven in one HTTP submission. Request count is not a monetary estimate. Review
provider pricing, actual account usage, the scientific protocol and a small
pilot before collecting a new experiment. Existing ten-study pilots are
preserved; this notebook uses full runs rather than resampling their data.

Saved partial or interrupted archives are reused **as partial results**, not
automatically resumed. Do not delete an archive or choose a new run ID to bypass
recovery. The original runner remains available for reviewed recovery with its
strict input/code/environment fingerprints. If a source change prevents live
resume, investigate it rather than relaxing that validation.

**“A prior request has an unknown outcome”** means the provider may have
processed a request without returning a response. Review pending state, receipts
and provider activity. If a study is deliberately skipped, retain it as unscored
with its original unknown-outcome record and recovery audit; a local terminal
failure is not a screening exclusion or proof of provider failure. A missing
prediction with a successful raw receipt may be recovered offline without
resubmission. A completion status alone does not establish full coverage.

Textless records remain in the cohort and manual-review queue. A title-only
study can still be scored. Both model and approach result directories are
separate. Existing raw responses, cohort/reference files and provenance are
never rewritten by comparison. See the [reproducibility guide](docs/reproducibility.md)
and [Inference.sh guide](docs/inference_sh.md).

## 5. Interpret the evaluation in screening terms

Notebook 04 compares saved probabilities with the separate human reference CSV.
The positive class is **human retain for full-text assessment**. These decisions
are the comparison reference, not a guarantee of final eligibility. Worksheet
discrepancies need review, not automatic relabelling.

A score becomes a screening decision only after choosing a **cutoff**: retain
when the score is greater than or equal to it. Lowering a cutoff generally
retains more studies, reducing missed reference-retain studies while increasing
the review queue. A cutoff of 0.5 need not achieve 95% recall.

| Readout | What it means here |
| --- | --- |
| Recall / sensitivity | Of human-retain studies, what fraction did the model retain? Missing these risks removing relevant evidence early |
| Precision / PPV | Of model-retained scored studies, what fraction did humans retain? Lower precision means more additional citations to assess |
| Specificity | Of human-exclude scored studies, what fraction did the model exclude? |
| F2 | Combines precision and recall, giving greater weight to recall |
| Confusion matrix | Counts true/false positives and negatives at a stated cutoff; unscored studies stay outside the matrix |
| ROC AUC | Describes ranking of human-retain versus human-exclude studies across cutoffs |
| PR AUC / average precision | Describe precision–recall behaviour; trapezoidal PR AUC and non-interpolated AP are reported separately |
| WSS@95 | The current implementation's excluded fraction minus 0.05, reported only where the stated recall scope reaches 95% |

Retaining 95 of 100 human-retain studies gives 95% recall, but does not tell you
how many human-exclude studies were also retained. Report precision and workload
together with recall.

Distinguish **scored-study** from **full-cohort** metrics. Dropping unscored
human-retain studies can make recall appear better. The notebook reports:

- Recall among scored studies.
- A conservative full-cohort recall lower bound, giving no credit for unscored
  studies.
- A separate scenario where all unscored studies are manually retained.

Unscored studies remain in the manual-review queue; they are not given zero
probabilities. Cohort WSS keeps them in that queue. For example, excluding 80%
of the cohort while meeting the recall requirement gives WSS@95 of 75 percentage
points under this implementation. It does not measure staff time or full-text
reading effort. The supplied protocol also contains a rank-based definition;
reconcile the difference before final reporting.

### Find the readouts and choose thresholds carefully

Notebook 04 **Sections 1–4** review all four sources, reference data, coverage
and exact archived questions. **Section 5** compares AUC on available scores
and the identical all-four-scored subset. **Sections 6–7** present the four-arm
recall/precision/specificity/F2/WSS readouts, ROC/PR curves and confusion matrices.
**Section 8** keeps hybrid R2/R3 secondary, and **Section 9** audits paired missed
reference-retain studies across all six arm pairs. Notebook 05 presents the
same four primary arms together; no approach is hidden by a default switch.

Each arm gets its own exploratory 95%-target cutoff and a 0.5 baseline. No
configured official threshold is adopted by these readouts. Preserve exact
cutoffs when ties affect recall. Different archived wording and missing scores
must be reported alongside apparent performance differences.

Selecting and assessing a cutoff on the same labels gives an **exploratory
estimate**, not independent validation. Formal hybrid tuning requires the
original development/tuning membership, reconciliation of the protocol's
7,000 records with the prepared 7,008-record cohort, and outstanding protocol
decisions. Do not invent a replacement split. Keep the 2,000 unseen studies
separate until the model, prompts, combination rule, threshold and missingness
policy are reviewed and locked. Reaching 95% here does not guarantee 95% on new
studies.

## 6. Export a report with code for colleagues

Run this terminal command from the repository root. It executes notebook 05's
saved-result summary and writes HTML under ignored `results/`, with code shown.
It needs the private artifacts and makes **no model API calls**. Keep the optional
`EXPORT_INTERIM_REPORT` flag false unless you also want its separate export.

```sh
uv run --locked --group notebooks jupyter nbconvert notebooks/05_study_summary.ipynb --execute --to html --embed-images --TemplateExporter.exclude_input=False --ExecutePreprocessor.timeout=180 --output 05_study_summary_with_code --output-dir results/notebooks
```

Open `results/notebooks/05_study_summary_with_code.html` in a browser, then
**Print → Save as PDF**. Landscape can help wide tables. This needs no XeLaTeX.
If all outputs are already saved in the notebook, omit `--execute` to export
those outputs without recalculation. Do not apply execution/export commands
indiscriminately to notebook 03: execution can submit paid requests.

For an aggregate report without code, use notebook 05 Section 10. Enable
`allow_writes` in `configs/evaluations/interim_summary.yaml` and
`EXPORT_INTERIM_REPORT` in the notebook. Notebook 04 Section 10 provides a
separate private four-arm export controlled by `configs/evaluations/comparison.yaml`.
Review outputs before sharing: executed notebooks can contain private material.

## 7. Keep the research record and private material separate

```text
jev-depression-evaluation/
├── README.md                 # Getting started and methodological context
├── pyproject.toml            # Package requirements
├── uv.lock                   # Resolved dependency versions
├── .env.example              # Empty credential template
├── configs/                  # Version-controlled experiment settings
├── prompts/reviewed/v1/      # Reviewed screening questions
├── notebooks/                # Explanations and executable workflow
├── src/jev_eval/              # Reusable Python functions
│   └── backends/              # Hosted and future local adapters
├── tests/                    # Synthetic data and mocked API checks
├── docs/                     # Methods and reproducibility documentation
├── data/examples/            # Invented examples retained in Git
├── data/private/             # Real prepared inputs: ignored by Git
├── results/                  # Outputs and recovery audits: ignored
├── .env                      # Credentials: ignored
└── .venv/                    # Local Python environment: ignored
```

Manifests record data/question/configuration identities, environment and code
checksums, model IDs and timestamps. Receipts preserve the available response
and provenance. A fixed sampling seed helps repeat a sample; it does not
guarantee identical hosted responses. An app name is not an immutable model-weight
revision. Keep that revision null unless reported, and do not claim it is
verified from a model name alone.

Install the commit guard once per clone:

```sh
uv run --locked --group notebooks pre-commit install
```

Before committing, save any executed copy privately under `results/notebooks/`.
Clear **all outputs and execution counts** from the source notebook and save it;
Jupyter's command palette has a clear-all-outputs command. Check what Git sees:

```sh
git status --short
uv run --locked --group notebooks python scripts/check_repository.py --working
```

The hook and GitHub checks reject notebook outputs, protected paths and likely
credentials. `.gitignore` excludes real data, credentials, weights, results and
local environments; it does not remove files already tracked by Git. Do not
force-add ignored files or put real citations in synthetic examples. Checks
cannot recognise every sensitive passage, so review staged changes before
sharing them.

## 8. Common problems

| What you see | What to do |
| --- | --- |
| `uv: command not found` | Install uv, reopen the terminal and check its version |
| No `pyproject.toml` found | Open the terminal in the repository root |
| GitHub “repository not found” | Check the URL, signed-in account and private-repository access |
| `ModuleNotFoundError: jev_eval` or missing packages | Run locked sync, select the project `.venv` kernel and restart it |
| Cannot import a recently added function | Restart the kernel; it can retain earlier module definitions |
| Missing workbook, prepared JSONL or results | Obtain private artifacts and check YAML paths; Git does not supply real data |
| Output still says `demo` | Check the loaded profile or `DATA_MODE`; `configs/notebooks.yaml` is not the full-run switch |
| Authentication error | Check `INFERENCE_API_KEY` privately and restart the kernel; never print its value |
| Live execution blocked | Inspect the specific flag, YAML permissions, dry-run policy, fingerprints or pending state named in the message |
| Unknown request outcome | Review receipts/provider activity and record a recovery decision before resuming |
| A candidate matrix is unavailable | The chosen recall target may be unattainable with available scores; inspect missing positives and coverage |
| `xelatex` not found | Export HTML and use browser Print to PDF |
| Malformed `.dist-info` version metadata | Stop changing the environment and have conflicting metadata inspected; do not alter the scientific package version to hide the error |

## 9. Optional developer checks and further reading

These checks use invented records and mocked HTTP, not paid calls:

```sh
uv run --locked --group notebooks pytest
uv run --locked --group notebooks ruff check src tests scripts
uv run --locked --group notebooks ruff format --check src tests scripts
```

Use the [notebook guide](notebooks/README.md) for detailed execution settings,
the [reproducibility guide](docs/reproducibility.md) for provenance,
the [methodological decision log](docs/methodological_decisions.md) for reviewed
choices, and the [Inference.sh guide](docs/inference_sh.md) for backend details.

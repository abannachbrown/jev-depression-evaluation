# Research notebook workflow

Run the notebooks in order the first time. Each notebook also works independently from a fresh kernel. Notebook 03 loads
both real pilot profiles and has independent execution flags currently enabled at the user's request. Notebook 04 reads the completed full runs and prepared human CSV by default;
its two-study demo now requires an explicit DATA_MODE change.

| Notebook | Purpose | Default side effects |
|---|---|---|
| 00_setup | Check the environment, settings, and reproducibility | None |
| 01_prepare_data | Load and validate JSON citation states | None |
| 02_model_connection | Test one selected JEV model through the Inference.sh REST contract | Depends on connection YAML; live calls incur charges and save results |
| 03_run_models | Run the same ten studies through both JEV models | Private sample manifest; both model execution flags are currently enabled |
| 04_evaluation | Compare decisions, coverage, and screening metrics | In-memory tables and figures only |
| 05_study_summary | Interim development-cohort report for colleague review | In-memory aggregate tables/figures; optional private export |

## Open the notebooks

From the repository root:

```sh
uv sync --locked --group notebooks
uv run --locked --group notebooks jupyter lab
```

Use the Python kernel from this repository's `.venv`. In an IDE, select that
interpreter explicitly. Notebook 00 checks the kernel and package versions.
Dependencies are managed in the terminal with uv, never installed by cells.

## Configuration and permissions

The shared starting point is `configs/notebooks.yaml`. Its evaluation profile
selects the dataset, models, generation settings, and backend configuration.
Relative paths resolve from the repository root, not the notebook directory.
Optional empty directory environment overrides fall back to the configured path.

The shared YAML defaults are synthetic data, demo mode, no paid calls, no writes,
and no display of citation text or raw responses. Notebook 03 explicitly selects
both hosted pilot profiles instead, with sample/result writing enabled and
independent `RUN_JEV_9B` and `RUN_JEV_27B` flags currently on. Executing the model cells can incur charges. The demo backend creates invented
answers; it does not imitate either JEV model. The evaluation fixture is
handcrafted separately for teaching the metrics.

Notebook 02 uses its own `configs/connections/inference_sh.yaml`, leaving the
other notebooks' demo settings unchanged. Select JEV-9B or JEV-27B v1.1 with
`model_config_file`. Mock mode makes no external requests; the current connection
configuration selects live mode independently of the other notebooks.
Live connections need `execution.mode: live`, `execution.allow_paid_calls: true`,
`INFERENCE_API_KEY`, and `ALLOW_PAID_CALLS = True` in the cell. There is exactly
one state/question request, without retries or polling. See the
[provider guide](../docs/inference_sh.md). Notebook 03 supports a separately enabled ten-state Inference.sh pilot.
Unconfigured Inference.sh batches remain blocked; generic hosted providers still
require an adapter.

Exports require `output.allow_writes` in YAML and explicit write permission in
the relevant cell. Notebook 03 passes `allow_writes=True` in each model call
because persistence is mandatory for the pilot; each separate `RUN_JEV_*` flag
controls whether its paid call executes. Generated files are restricted to
`results/` inside this repository.

For a full approximately 5,000-state/two-model run, configure the private data,
set `max_records: null`, and deliberately raise request/attempt limits after a
small verified test. The initial logical-request limit of 10 prevents an
accidental full batch. The runner is sequential (`max_concurrency: 1`).

## Save and share safely

Install the repository's commit hook in each clone:

```sh
uv run --locked pre-commit install
```

Before committing, use Jupyter's **Clear Outputs of All Cells**, save, and stage
the clean notebook. The hook inspects the staged Git blobs. It rejects outputs,
execution counts, attachments, extra execution/widget metadata, protected file
locations, and likely secrets. It never silently changes or stages files.
CI repeats those checks against the committed files.

Keep local executed notebook copies under `results/notebooks/`, which is ignored.
Do not paste real records, credentials, or response excerpts into Markdown or code
cells. Output checks cannot remove source text, and secret scanners cannot detect
every sensitive passage. Hooks can be bypassed; CI does not erase leaked history.

For an unstaged working-tree review:

```sh
uv run --locked python scripts/check_repository.py --working
```

## Verify execution

```sh
uv run --locked --group notebooks pytest
```

The notebook tests launch fresh kernels, forbid the package's HTTP transport,
use only synthetic defaults with permissions off, and keep execution output in
memory. They never overwrite source notebooks. Notebook 02 permits mocked HTTP
transport, while actual HTTP transport is blocked. Existing non-synthetic workflow
settings intentionally fail the test prerequisite check rather than being run.

No notebooks train or fine-tune models, install GPU runtimes, or download weights.

## Interim colleague-review report

Open `05_study_summary.ipynb` independently in a fresh project `.venv` kernel and
use Run All **in notebook 05**. It reads the saved full runs without executing
notebooks 02 or 03, constructing an API client, loading `.env`, or requiring an
API key. Missing scientific inputs raise a prerequisite error instead of
silently selecting demo data.

`configs/evaluations/interim_summary.yaml` selects the existing analysis profile,
exploratory recall target and illustrative model cutoffs. It records validation
on 2,000 unseen studies as **planned after colleague review**, not performed.
It does not change `analysis_full.yaml`, model configurations or live execution
flags. The report contains data-quality/audit findings, archived criteria,
coverage, model metadata, exploratory metrics/figures, aggregate paired errors,
provenance and an author-review methods draft. Unknown human-adjudication
procedures and immutable hosted weight revisions are not inferred.

The default run writes nothing. An optional self-contained HTML/CSV/PNG/PDF
bundle requires `allow_writes: true` in the summary YAML and
`EXPORT_INTERIM_REPORT = True` in Section 10. New report bundles go under ignored
`results/summary/interim/`, with no citation text, study/ITEM_ID values, raw
responses or credentials. Source notebooks retain empty outputs for Git.
Review the aggregate report before sharing it with colleagues. Cutoffs and the
validation protocol should be locked after review and before inspecting the
unseen validation outcomes.

## Ten-state pilot execution

Notebook 03 contains explicit code for **both** models. Keep the shared
`configs/notebooks.yaml` demonstration settings unchanged.

1. **Sections 1–3:** load both pilot profiles, freeze one ordered ten-study sample,
   and review both plans and the shared eligibility instructions. These sections
   make no API calls; sample preparation can write a private manifest.
2. **Section 4 — potentially billable:** enable `RUN_JEV_9B` to execute the
   JEV-9B call, then inspect its status in **Section 5**.
3. **Section 6 — potentially billable:** independently enable `RUN_JEV_27B` to
   execute JEV-27B v1.1 on the same ten states. Inspect it in **Section 7**.
4. **Sections 8–9:** view the study-ID-based paired probability table and a
   generated methods description/provenance table for paper preparation.

No workflow path needs to be switched or resampled between model cells. Both
execution flags are currently true; executing either model cell can incur charges. Separate `result_9b` and `result_27b` objects
and run directories preserve model identity. Reporting shows absent, failed and
unknown responses explicitly. The methods draft only claims completion when
both results contain ten successes and completed status, without pending state.

Use **Section 11** for resume guidance and the optional **Section 10** for raw
inspection. The runner stops on its first error; unknown outcomes require review
before resume. Saved successes and known terminal failures are not automatically
submitted again. Notebook 02's one-state safeguards are unchanged. See
[the pilot guide](../docs/inference_sh.md#ten-state-pilot-in-notebook-03).

Keep IDs, outputs and executed notebooks private. Clear outputs before Git.
Notebook 04's metrics require a separately specified probability threshold and
a documented threshold. Its full CSV reference loading is now implemented.
The reporting functions make no API calls.

## Full cohort workflow — appended sections 12–20

Start notebook 03 at **Section 12 from a fresh kernel**. The original 22 pilot
cells are preserved and their earlier execution flags are enabled; do not use
Run All to enter the full workflow.

- **12–15:** load both dedicated full profiles, validate the complete JSONL and
  separate human-reference CSV, inspect exact eligibility/configuration checksums
  and review the dry plan. Counts are determined from the data. No API calls or
  output writes are made by these sections.
- **16:** explicitly enable `RUN_FULL_JEV_9B` only after disabling dry_run and
  enabling paid/write permissions in `configs/workflows/full_jev_9b.yaml`.
- **17:** separately enable `RUN_FULL_JEV_27B` after the corresponding YAML changes
  and review. Both models use every identical source state and the same question;
  records without any text are explicitly retained without submission.
- **18–19:** read saved validation and reproducibility metadata without resuming.
- **20:** create the row-level reference join in memory. Export requires
  `EXPORT_FULL_RESULTS = True` and YAML write permissions. Missing, failed,
  unknown and textless rows are preserved, with no performance metrics here.

All full execution/export flags start false. To reload changed settings, rerun
12–15; the dry run does not transition into live inference. Full results are
separate under `results/full/depression/full-depression-v1/`. The current cohort
plans 13,994 initial submissions across models, plus bounded 429 retries; the
request ceiling is not a monetary spending cap. See the
[full-run protocol](../docs/inference_sh.md#full-evaluation-in-appended-notebook-sections-1220)
for recovery, nullable revisions, validation, privacy and configuration limits.

## Evaluate saved full runs in notebook 04

Notebook 04 defaults to `DATA_MODE = "full"` and
`configs/evaluations/analysis_full.yaml`. It loads the complete prepared cohort,
human-reference CSV and both saved full model runs. It never resumes inference,
constructs a backend or needs an API key. The shared `configs/notebooks.yaml`
remains the separate two-study demo and does not choose full-analysis inputs.

Restart the kernel and run notebook 04 from the top. Section 2 should report
7,008 human reference-labelled studies for the current data. The probability
coverage is 6,997 per model, with eleven textless studies retained explicitly.
Counts are calculated from the files and not hardcoded.

The user has left `decision_threshold: null`. Coverage is available immediately;
binary metrics and confusion matrices remain disabled until a threshold is
explicitly chosen and documented in the analysis YAML. Retention uses
`P(true) >= threshold`. With incomplete labelled coverage, headline metrics stay
withheld by default. `report_partial_metrics: true` explicitly permits metrics
conditional on valid labelled pairs; it never removes the missingness counts.

The original inference fingerprints, code and environment are preserved when
loading archives. Changes to evaluation code do not require another model call.
Source/cohort/task identities, model selection, duplicate IDs, probabilities and
saved-output integrity are still checked. Both inference runs and their
provenance remain read-only.

Optional exports require `allow_writes: true` in the analysis YAML and
`ALLOW_WRITES = True` in the notebook. Unique private bundles record coverage,
available threshold-based summaries/figures, threshold and partial-coverage
settings, input checksums and separate inference/analysis identities. Keep
executed notebooks and results outside Git; clear source outputs before commit.

For teaching, explicitly set `DATA_MODE = "demo"`; its two invented references
and figures remain labelled synthetic. Automated checks cover both modes and
forbid inference in the evaluation notebook.

## Exploratory recall-target readouts

Notebook 04 Sections 8–9 now examine observed cutoffs using saved probabilities,
without changing `decision_threshold` or making API calls. The highest feasible
observed cutoff is selected separately per model, using inclusive `P >= cutoff`
and exact positive-score ranks/ties. The default target is 95% recall with all
human-retain labels in the denominator and no credit for unavailable scores.
Scored-only candidates, the 0.5 baseline and 95/97/99% targets are also displayed.

The current reference has 957 retain labels, of which 946 have a model score;
all eleven textless studies are reference-retain. This makes the denominator
important. Recall across all labels is a conservative lower bound, not a negative
prediction assigned to the unavailable studies. All unscored records should stay
in a separate manual/retention queue. A separate scenario reports what would
happen if all unscored records were retained; it is not model-only performance.

The plots show recall/workload tradeoffs and explicitly exploratory scored-pair
confusion matrices even while the official threshold remains unset. No model
threshold is automatically selected for the configured analysis. Exact cutoff
values are displayed at full precision; rounding upward can drop tied positives.
Queue counts include manual records and automatic exclusions include false
negatives. The 95% target is an observed point estimate on this cohort, not a
confidence guarantee or held-out validation. Lock and validate chosen cutoffs
on independent data before claiming generalisation.

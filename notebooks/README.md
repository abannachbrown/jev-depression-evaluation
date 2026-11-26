# Research notebook workflow

Notebooks 03–05 present the **two models × two NOUL designs** together:

| NOUL design | JEV-9B | JEV-27B v1.1 |
| --- | --- | --- |
| Binary | Original saved binary run | Original saved binary run |
| Hybrid | Saved hybrid-v1 run | Saved hybrid-v1 run |

The exact archive sources are configured in
[`configs/evaluations/comparison.yaml`](../configs/evaluations/comparison.yaml).
This selection is independent of `default_approach` in the setup screening YAML.
There is no automatic fallback between designs. Missing arms and unavailable
scores remain visible rather than being replaced with another experiment.

**No paid responses need to be repeated to generate this comparison.** The four
execution flags in notebook 03 are false. Existing archives are reused even if
a flag is enabled. Notebooks 04 and 05 contain no inference calls.

## 1. Open the notebooks

From the repository root, using the existing uv environment:

```sh
uv sync --locked --group notebooks
uv run --locked --group notebooks jupyter lab
```

Select the project `.venv` kernel and restart it after source changes. Read
Markdown before running a code cell with Shift + Enter. See the
[main README](../README.md) for installation and beginner instructions.

| Notebook | Current role | API calls |
| --- | --- | --- |
| 00_setup | Shared environment and four-arm configuration/availability table | None |
| 01_prepare_data | Verify the same ordered cohort and mapped reference across all four arms | None |
| 02_model_connection | Four labelled mock examples, original individual tests and saved-connection inventory | Mock by default; live only through explicit individual permissions/flags |
| 03_run_models | Explicit four-arm archive review/reuse and separate optional execution cells | Defaults off; existing archives never resubmitted here |
| 04_evaluation | Side-by-side coverage, metrics, curves, matrices and paired audit | None |
| 05_study_summary | Four-arm interim methods/results report for colleague review | None |

Code remains in reusable modules under `src/jev_eval/`. No notebooks install
packages, download weights or add local GPU, training or fine-tuning workflows.

## 2. Preserve the common data and original experiments

Each state has only `study_id`, `title` and `abstract`. Human inclusion status and
ITEM_ID remain in the separate reference CSV, linked by original string study ID.
Manual psychiatric-context flags stay in separate private annotations; they
are not an eighth NOUL or an exclusion rule.

Notebook 01 reuses the Excel conversion without modifying its workbook. Missing
abstracts are warnings. Title-only studies can be scored; entirely textless
studies stay in the cohort as unscored and require no API submission.

Notebook 00 checks all four configurations without reading private datasets or
constructing clients. Its table distinguishes configured YAML permissions from
actual execution and reports unavailable private files without fabricating them.
Notebook 01 defaults to `DATA_MODE = "prepared"`: it reads all four source
selections and blocks different citation content/order or mapped human reference
decisions. It retains one shared input cohort/reference and previews one sample
without freezing or overwriting pilot manifests. Its demo mode uses invented
records and explicitly does not verify real-data equivalence.

Four-arm loading verifies the identical ordered citation cohort and reference
data across both designs. Each archive retains its actual task and inference
fingerprint. **The original binary wording differs from hybrid-v1 wording.**
These saved experiments can be compared descriptively, but differences cannot
be attributed solely to question design. The report prints that limitation and
checks the definitions. It never relabels the old binary run as a new v1 run.

The reviewed binary-v1 templates remain available separately; selecting those
for a future controlled experiment would be new inference, not reuse of the
archived binary responses. No such new experiment is enabled by this change.

## 3. Notebook 03: see both approaches on both models

Run from the top:

1. **Section 1:** load four explicit source identities and statuses.
2. **Section 2:** inspect cohort/reference linkage and actual archived criteria.
3. **Section 3:** review saved counts, exceptions, timestamps and token usage.
4. **Section 4:** four individually labelled binary/hybrid × model cells. All
   flags are false. Existing arms return `reuse_saved_run` without a client or
   POST, including incomplete arms.
5. **Section 5:** reload coverage and proceed to evaluation.

`RUN_BINARY_JEV_9B`, `RUN_BINARY_JEV_27B`, `RUN_HYBRID_JEV_9B` and
`RUN_HYBRID_JEV_27B` are independent. For an absent arm only, deliberately
setting its flag true may submit paid requests and write files if the selected
workflow also permits it. It reuses the existing bounded runner, full-cohort
validation, rate limits, mandatory raw saving and strict resume safeguards.
Binary sends one NOUL per study; hybrid sends seven in the same POST. Pricing
must be checked with the provider rather than inferred from request count.

Existing directories are not resumed or overwritten through these comparison
cells. Resolve an incomplete original experiment using its reviewed original
workflow; a changed source/environment fingerprint must not be bypassed. An
unknown-outcome request can have completed remotely and must not be reposted
without review. Deliberate skip decisions preserve original records and audits.
A successful raw receipt missing from predictions may be recoverable offline.

The original ten-study pilot records and previous executed notebooks remain
private and unchanged. The four-arm comparison uses full archives, not a new
sample, and does not claim a partial run was completed by being reused.

## 4. Notebook 04: compare saved outputs

| Section | Readout |
| --- | --- |
| 1–2 | Four source identities, same cohort/reference, wording comparability |
| 3 | Successful, failed, unknown, textless and missing records for every arm |
| 4 | All saved question-score summaries and exact archived criteria |
| 5 | ROC AUC, trapezoidal PR AUC and AP; per-arm available scores and common scored subset |
| 6 | Arm-specific exploratory recall-target cutoffs, 0.5 baseline, PPV, specificity, F2 and WSS |
| 7 | Four-arm ROC/PR and recall/workload curves, four candidate confusion matrices |
| 8 | Secondary hybrid R2 minimum/R3 mean and protocol status |
| 9 | Paired missed-positive audit for all six arm pairs |
| 10 | Optional private aggregate comparison export |

The main hybrid score is `overall_eligibility`; R2/R3 do not become extra primary
arms. They combine the first four domains: primary research, animal study,
in vivo experiment and depression model. Control/outcome questions are secondary.
Derived scores are not demonstrated calibrated joint probabilities. R1 stays
disabled pending protocol clarification.

The cutoff comparison selects each arm's highest observed cutoff achieving
95% recall over all human-retain studies, giving unavailable predictions no
model credit. Equality/ties are included. If unattainable, the status remains
explicit. No official threshold is changed. Precision, specificity, F2 and
confusion counts use scored labelled pairs. Unscored studies remain in the
manual-review queue. WSS@95 uses excluded fraction minus 0.05 where recall meets
95%; the protocol's alternative rank-based definition still requires review.

Available-score AUC denominators can differ. Common-scored AUC additionally
restricts all four arms to the identical set of studies with scores in every
arm. Neither table establishes independent validation, calibration or a causal
effect of question design. Paired errors are descriptive, not a significance
test. Models without both reference classes receive unavailable AUCs.

Formal rule tuning still requires the original 5,749/1,251 membership and
reconciliation with the 7,008-record cohort. Do not create a replacement split
or drop eight records silently. The planned 2,000 unseen studies are not loaded
or used for these exploratory thresholds.

## 5. Notebook 05: interim report covering all four arms

The summary YAML selects `comparison_config`. All primary flow, execution,
performance, AUC, curves and matrices include the four arms. Wording caveats,
missingness and paired errors accompany the readouts. Exact archived questions,
workbook audit findings, original inference provenance and a methods draft are
included. Secondary hybrid rules remain separately labelled.

Colleague review is pending and independent validation is planned. This report
does not approve either stage or automatically lock a model/rule/cutoff.

## 6. Connection safeguards remain separate

Notebook 02's original connection remains one synthetic state, one question and
one POST. Its walkthrough now selects the dedicated `comparison_binary.yaml`
mock profile; the original live YAML is unchanged and the paid cell flag is
false. The separate hybrid connection profile,
`configs/connections/hybrid_v1.yaml`, defaults to mock mode, seven questions and
one POST, with paid/writes off and its notebook flag false. Fabricated mock
probabilities demonstrate the pipeline, not model accuracy.

A deliberate live test needs `INFERENCE_API_KEY`, the selected connection's
live mode, YAML paid/write permissions and its own notebook flag. Credentials
are loaded from the environment/ignored `.env`, never printed or committed.
Successful live responses save automatically; a storage failure calls for
save-only recovery rather than another paid request.

Section 9 uses `configs/connections/comparison.yaml` to show separate binary and
hybrid examples for each model, with a shared synthetic state. The loader rejects
live mode and paid/write permissions in these profiles: the examples are
explicitly mock-only, with fabricated probabilities. Each retains its own
one-state, one-POST limit and either one or seven questions. It reuses the
existing request builder, client and response validation.

Section 10 reads only an allowlist of saved connection metadata fields and
reports whether normalized/raw files exist. It does not resubmit, validate or
recover earlier tasks; nonstandard historical recovery folders remain intact.
Connection examples are not evaluation data. All three updated source notebooks
have private backups of their earlier executed versions under results/notebooks.

## 7. Export, share and verify

Notebook 04 Section 10 requires `allow_writes: true` in the comparison YAML and
`EXPORT_COMPARISON = True`. Notebook 05 Section 10 requires summary-YAML writes
and `EXPORT_INTERIM_REPORT = True`. New private folders contain aggregate
HTML/CSV/figures and manifests identifying all four original runs, reference,
source/configuration hashes and analysis code. Inference files stay untouched.

For a printout with code, use the main README's notebook-05 HTML export, then
browser Print to PDF. Do not bulk-execute arbitrary notebooks for export. Keep
executed notebooks, tables, figures, raw artifacts and reports under ignored
`results/`; private inputs remain under ignored `data/private/`.

```sh
uv run --locked pre-commit install
uv run --locked --group notebooks pytest
uv run --locked ruff check src tests scripts notebooks
uv run --locked ruff format --check src tests scripts notebooks
```

Clear source notebook outputs and execution counts before committing; preserve
an executed copy privately first. Tests use invented inputs and mocked HTTP,
including fresh-kernel four-arm execution with inference explicitly blocked.

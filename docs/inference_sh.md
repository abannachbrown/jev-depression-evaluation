# Inference.sh single-request connection

The backend implements the shared one-attempt `predict` / `close` interface for
NOUL, choice and score questions. The original connection sends one state and
one question. A separate reviewed hybrid profile explicitly allows one state
with seven NOULs. Notebook 03 supports paired pilot/full v1 workflows; see the
[current notebook guide](../notebooks/README.md) for execution flags and profiles.

## Configuration

For the current hybrid workflow use `configs/screening/v1.yaml`, the separate
`configs/connections/hybrid_v1.yaml` mock-first connection and
`configs/backends/inference_sh_hybrid.yaml` with `max_questions: 7`. The default
backend limit remains one. The seven IDs are fixed to the reviewed screening
domains plus `overall_eligibility`; all must be present in a successful response.
The builder collects every NOUL in the array rather than replacing earlier
questions. Original single-question connection limits have not been relaxed.

V1 pilot and full paths are separate for binary/hybrid and for both models.
New permissions and live flags start off, full scope starts in dry-run mode,
and partial responses retain their raw bodies with an error instead of counting
as success. R1 is disabled pending protocol review. R2/minimum and R3/mean are
offline score transformations, not additional model requests or calibrated joint
probabilities. Formal tuning needs the original reconciled tuning membership.

The sections below also document the historical binary implementation and
archived experiment locations; their old defaults/paths are not the v1 defaults.

Start in `notebooks/02_model_connection.ipynb`. No new dependencies are needed.
Connection settings are in `configs/connections/inference_sh.yaml`:

- `model_config_file`: `configs/models/jev-9b.yaml` selects `autotrust/jev-9b`;
  `configs/models/jev-27b.yaml` selects `autotrust/jev-27b-v1-1`.
- `state_file`: restricted to the existing checked-in synthetic citation state.
- `task_config`: selects one question from a validated request template.
- `execution.mode`: the current workspace selects `live`; use `mock` for offline
  checks. Live calls need explicit paid-call permissions.
- `screening.binary_threshold`: null means probabilities only. A threshold
  applies `P(true) >= threshold`; this is not a recommended screening cutoff.
- `output.allow_writes`: currently true, as required for live calls. Exports are
  restricted to `results/`.

`model_config_file` avoids Pydantic's reserved `model_config` attribute. Existing
generic hosted model-ID environment variables remain available; Inference.sh
uses each model YAML's `hosted.app_id`.

Backend settings fix the origin to `https://api.inference.sh`, path to `/apps/run`,
`wait` to true, and attempts to one. Timeout is configurable. No temperature,
output-token limit, or sampling seed is sent to this decision endpoint.

## Authentication and execution

Supply `INFERENCE_API_KEY` through the environment or ignored `.env`. Never put
the value in YAML or notebook source. Initialization loads `.env` without
replacing existing environment values and never prints credentials.

Live execution needs both `execution.allow_paid_calls: true` in connection YAML
and `allow_paid_calls=True` in the call. It submits exactly one POST. There are
no automatic retries, redirects, polling, or streaming. A timeout can mean the
provider already accepted the task: do not automatically resubmit. Investigate
any returned task ID in the provider dashboard.

Mock mode uses `httpx.MockTransport`, a dummy key, and the handcrafted
`data/examples/inference_sh_response.json`. Its probability and usage are invented.
The returned model name is adjusted to the selected test app; neither the fixture
nor mock latency measures actual model performance. The fixture covers the default
binary question; a different question needs a corresponding mock response.
Choice and score inputs/outputs are exercised by provider unit tests.

## Questions and normalized answers

Task templates have a `questions` mapping with typed entries:

| Type | Input | Normalized output |
|---|---|---|
| `noul` | Instructions; optional true/false criteria | `probabilities[id]`: probability of true |
| `choice` | Instructions; uniquely named options, optional descriptions | `choices[id]`: choice, confidence, option probabilities |
| `score` | Instructions; ordered levels | `scores[id]`: weighted score, normalized score, confidence, probabilities, legend |

Question IDs and option/level keys must match the request. Probabilities are
finite numeric values within 0–1; strings and Booleans are rejected. Choice/score
distributions sum to one within absolute tolerance 0.00001. Independent NOULs
are not summed. Score indices start at zero, and the weighted score may be
fractional rather than an integer selected level.

HTTP success alone is insufficient. The parser requires `data.id`, an integer
`data.status`, and status 10 before validating output. Statuses 11/12/13 mean
failed/cancelled/rejected. Other unfinished statuses stop without extraction.
Unrequested output maps may be absent; requested IDs must be present. Additional
task metadata remains in the raw response. Fields used for extraction are typed.

Requested app and returned `output.model` are recorded separately. JEV-27B v1.1
may return `autotrust/autotrust-jev-27b-v1.1`. A name is not a pinned weight
revision. Only available input-token usage is recorded; output-token counts and
monetary cost are not inferred.

## Raw responses and exports

The backend captures response bytes, status, receipt time, and elapsed time
before parsing. Sanitized errors retain the private exchange, including HTTP
errors and malformed JSON. Without a received HTTP response there are no raw
bytes. The connection helper returns a failed result with the available exchange.

Live calls always save automatically. Set `output.allow_writes: true` in the
connection YAML; live calls stop before sending if saving is disabled or the
output folder is not writable. Mock mode writes only when both YAML write
permission and `allow_writes=True` are provided. Each export uses a new
directory with exact `response.body` bytes when received, `request.json`,
`metadata.json`, and `normalized.json` only after successful validation. Metadata
includes execution mode, app/model IDs, task ID, timestamps, state/task/request
hashes, settings, Git state, package versions, lockfile hash, and SHA-256 hashes
of the executing Python package source files. Source hashes identify code even
when it has not been committed. No authentication
headers, cookies, or credential values are exported.

The notebook prints the saved directory. If storage fails after a request,
`ConnectionPersistenceError.result` preserves its response and metadata in memory.
The notebook retains this as `result`. Fix storage and run the separate save-only
recovery cell; do not rerun the paid request cell. A writable-folder check cannot
guarantee that storage will remain available after the API call.

The earlier JEV-9B result was recovered from notebook stdout under
`results/connections/recovered-jev-9b-5ca7hfm4xt0r1e18e3e17ey31s/`.
It includes a notebook snapshot and explicitly incomplete recovery metadata.
Its formatted JSON is not the original HTTP bytes, and missing original client
provenance has not been reconstructed from current settings.

Keep live responses under ignored `results/`. Never replace the synthetic fixture
with a live response. Clear notebook outputs before committing; existing hooks
and CI enforce this. Tests use mock transports and synthetic credentials only.
Notebook tests block real HTTP transport while permitting in-memory mocks.

## Ten-state pilot in notebook 03

Notebook 03 explicitly loads both `configs/workflows/pilot_jev_9b.yaml` and
`configs/workflows/pilot_jev_27b.yaml`. There is no model-switching step or need
to edit YAML between runs. The profiles reuse the existing model YAMLs, prepared
7,008-record JSONL, seed 42 and `prompts/drafts/binary_noul.json`. The general
`configs/notebooks.yaml` demonstration settings remain separate.

Run the numbered notebook sections in order:

1. **Section 1:** load both profiles. No request is sent.
2. **Section 2:** freeze or verify the shared ten-study sample. With
   `ALLOW_SAMPLE_WRITES = True`, the first execution writes the private sample
   manifest, without contacting the provider.
3. **Section 3:** build `plan_9b` and `plan_27b` from the same `sample`, inspect
   both plans and the exact shared eligibility question. The package verifies
   identical sample/order/task/seed and separate output directories.
4. **Section 4 — billable:** set `RUN_JEV_9B = True` to run only JEV-9B.
5. **Section 5:** review `result_9b`, errors, coverage and provider charges.
6. **Section 6 — billable:** set `RUN_JEV_27B = True` to run only JEV-27B v1.1
   with the same sample; the JEV-9B call is not invoked by this cell.
7. **Sections 7–9:** inspect the second model, paired probabilities and a
   paper-oriented methods draft with provenance.

Both execution flags are currently true at the user's request. Running either
model cell or Run All can submit potentially billable requests. Result saving is
explicitly enabled in each call and required by the runner. Restart the kernel
after source changes. The cells have stable descriptive IDs for testing, while
section headings provide the execution instructions for researchers.

The probability table joins by original study ID in frozen sample order. It
shows separate statuses/probabilities and retains missingness rather than
substituting zero. It does not classify eligibility or calculate accuracy. The
methods helper derives sample counts, seed, app IDs, fingerprints, coverage and
available run metadata from the actual objects; it describes a completed pilot
only when both model results have ten successes, completed status and no pending
marker. Otherwise the draft is explicitly a planned protocol with incomplete
execution. A result absent from the kernel does not mean no saved run exists.
The provenance table records source/sample/task and lockfile checksums, Git
revision, returned models and timestamps. Review the draft before publication.

Section 10 optionally displays raw responses if explicitly enabled in YAML.
Keep executed notebooks and the paired ID/probability table private and clear
outputs before Git. Reporting functions make no API calls or file writes.

The sample helper records the checksum of source bytes, conversion-manifest checksum
and workbook checksum when available, ordered study IDs, normalized sample hash,
criteria hash, seed, validation issues and the IDs unavailable for inference.
Eleven records in the current prepared bundle have neither title nor abstract;
they remain in the dataset and human reference file, with no changed eligibility
labels. Ten records are selected from the 6,997 with available text. Title-only
records remain eligible for sampling. Duplicate IDs, unexpected decision fields
and other validation errors block preparation. Existing sample manifests are
never overwritten; source, seed or task changes require a separately named,
reviewed pilot rather than silently replacing an existing sample.

The runner permits only this explicitly enabled pilot: hosted mode, exactly the
frozen ten states, one supported app and one NOUL, raw-response saving, and at
most ten requests/attempts. It continues to reject unconfigured Inference.sh
batches. No automatic retry, polling, multi-state request or local inference is
introduced. The low-level backend's `supports_batch = False` continues to mean
that it sends one state per API request.

Each model writes to its own directory under `results/pilots/depression-10/`:
`jev-9b/` and `jev-27b-v1-1/`. Files include the shared-schema manifest,
`predictions.jsonl`, `responses.jsonl`, `failures.jsonl`, and `raw/` receipt JSON
plus exact HTTP `.body` files. Prediction rows include probabilities, requested
app, returned model, task/request ID, available input-token usage, HTTP status,
client timestamps and errors. Receipts link exact request/state/task/body hashes.
Text-generation settings required by the shared YAML schema are **not applied**;
the manifest records `generation_applied: false`. No probability cutoff or
human screening decision is inferred by this pilot.

The pilot requires writing, tests storage before its first POST, and checkpoints
before each attempt. Responses and terminal rows are flushed before an atomic
manifest update may clear pending state. A per-run exclusive lock prevents two
kernels from writing the same run. A killed process can leave a stale
`.<run-id>.lock` beside the run directory; inspect the process, receipts and
pending marker before manually removing it. Saved successes and known terminal
failures are not resubmitted on explicit resume. Changed scientific inputs or
source/lockfile fingerprints block resume.

The first error stops the pilot. Authentication, billing and throttling errors
are recorded as terminal failures; an explicit resume skips that failed record
and advances to unattempted states. Timeouts, unfinished tasks, malformed task
envelopes and server errors preserve pending state and block automatic resume.
Inspect any task ID in the provider dashboard. There is no automatic deletion
of pending markers or reposting. This is conservative recovery, not a server-side
exactly-once guarantee. Partial/corrupt checkpoint files also require review.

If storage fails, `PilotPersistenceError.result` retains the in-memory run and
its latest `_pending_exchange` when received. Notebook 03 keeps it as `result_9b` or `result_27b`.
Keep that kernel open, inspect receipts and the on-disk pending marker, and
recover storage without reposting. Local filesystem availability cannot be
guaranteed after a request has been accepted.

Request starts are spaced by at least 3.1 seconds, sequentially. Check your
account's actual rate/concurrency limits and other active clients before running.
A ten-request ceiling is not a monetary spending cap: input-token counts and
HTTP elapsed seconds are not billing measurements. Consult
[provider pricing](https://inference.sh/pricing) and the
[cost-estimate documentation](https://inference.sh/docs/api/rest/store#estimate-app-cost)
before scaling. Provider task costs can be reviewed by task ID via the dashboard
or the documented `GET /usage/tasks/:id/cost`; this workflow does not make those
additional API calls or estimate a dollar amount.

Notebook 04 now reads full saved outputs and the prepared reference CSV through
`configs/evaluations/analysis_full.yaml`. Boolean metrics require an explicitly
specified threshold; the user has currently left it unset. The workflow's
synthetic reference paths are unused by inference and are not real evaluation
labels. No new dependencies are needed; all pilot tests use invented data and
mock transports, including fresh-kernel checks with neither model, either model alone, and both
models in sequence. Re-executing a completed model cell is also checked to
produce no duplicate POSTs.

## Official references

- [JEV-27B v1.1 API](https://inference.sh/apps/autotrust/jev-27b-v1-1#api-reference)
- [JEV-9B API](https://inference.sh/apps/autotrust/jev-9b#api-reference)
- [Decision contract](https://inference.sh/docs/extend/app-code#decision-models-contract)
- [Authentication](https://inference.sh/docs/api/authentication)

No local GPU runtime, model weights, training, or fine-tuning are included.

## Full evaluation in appended notebook sections 12–20

Notebook 03 preserves its original 22 pilot cells and appends 18 full-evaluation
cells. Start at **Section 12 from a fresh kernel**; do not use Run All to enter
this workflow because the earlier pilot flags are enabled at the user's request.
The new `RUN_FULL_JEV_9B`, `RUN_FULL_JEV_27B` and `EXPORT_FULL_RESULTS` flags are
false. Neither implementation nor dry-run planning executes full inference.

The separate files are:

- `configs/workflows/full_jev_9b.yaml` and `full_jev_27b.yaml`;
- `configs/evaluations/full_jev_9b.yaml` and `full_jev_27b.yaml`;
- `configs/backends/inference_sh_full.yaml`.

Existing model, prompt, demo, pilot and one-state connection YAMLs are unchanged.
Both evaluation profiles select the complete prepared JSONL, `expected_count:
null`, and the same existing binary eligibility file. The shared schema's seed
and generation fields are unused for full decision inference: no sampling,
temperature, text-generation token cap or model seed is applied.

### Preparation and dry run

Sections 12–15 independently load both full profiles, validate every source row,
load reference CSV separately, display exact criteria/config checksums and build
two execution plans. No provider client is constructed and no output files are
created by planning. The actual count comes from the JSONL, not a hardcoded
7,008. Input fields must be exactly `study_id`, `title`, `abstract`; IDs remain
strings and extra decision/linkage fields are rejected.

The current source has 7,008 unique study IDs, 1,385 missing abstracts and eleven
records with no citation text. All studies remain in the cohort. The approved
`record_not_inferred` policy creates a `not_inferred_no_text` row for each of
those eleven in each model output, with null probability and zero attempts.
Title-only studies are submitted. A fresh pass therefore plans 6,997 submissions
per model, 13,994 combined, before retries; this is computed dynamically.
Existing ten-study pilot outputs are not automatically imported.

Initial full workflow settings are `dry_run: true`, paid calls disabled and
writes disabled. Other defaults are concurrency one, minimum 3.1 seconds between
starts, 120-second timeout, a 10,000 logical-request ceiling and a 12,000 total
attempt ceiling per model, at most three attempts per study, exponential 429
backoff from two seconds capped at sixty, and a pause after three consecutive
terminal study errors. Provider Retry-After can require a longer delay. Waits
are chunked so progress remains visible and execution can be interrupted.
The combined attempt ceiling is 24,000, not a dollar spending cap.

To start a reviewed model run, explicitly set that full workflow's
`inference_sh_full.dry_run: false`, `execution.allow_paid_calls: true` and
`output.allow_writes: true`. Rerun Sections 12–15 to reload/review the plan, then
enable only its full execution flag in **Section 16 (9B)** or **17 (27B v1.1)**.
Planning never enables those flags or changes YAML. Progress contains counts,
not study text or credential values. Inspect provider charges and actual account
limits before proceeding or scaling.

### Checkpoints, retries and recovery

The existing serial runner and Inference.sh backend are reused. The pilot
retains its exact ten-state gate and stop-on-error behavior; the one-state
connection remains one POST. Full scope is separately enabled, requires the
entire ordered source cohort and mandatory saving under `results/full/`, and
cannot be enabled simultaneously with pilot scope.

All full records share a frozen `cohort.json` with source byte/hash, ordered
normalized-state hash, exact IDs/order, conversion/workbook checksums where
available, task content/file checksums, missingness policy and validation audit.
Each model manifest references that file and keeps only compact cohort metadata,
avoiding rewriting thousands of IDs at every checkpoint. Paths are:

```text
results/full/depression/full-depression-v1/
├── cohort.json
├── full-jev-9b/
│   ├── manifest.json
│   ├── predictions.jsonl
│   ├── responses.jsonl
│   ├── failures.jsonl
│   └── raw/                 # exact bodies and per-attempt JSON receipts
├── full-jev-27b-v1-1/
│   └── ...
└── exports/<UTC-unique-id>/
    ├── evaluation_ready.csv
    └── manifest.json
```

Before a POST, a pending marker and attempt counter are durably written.
Responses/receipts and new JSONL rows are flushed before the pending marker can
be cleared atomically. JSONL files append only new records. Full runs keep
compact prediction rows and receipt references in memory; raw response bodies
and request content remain on disk. The latest exchange is retained privately
if storage fails. The CSV/reference files never become model inputs.

Only explicit HTTP 429 rejections are automatically retried. Retry-After is
retained by the transport, which itself still submits only once. The scheduler
records each failed attempt and persists its cooldown before clearing pending
state; retry limits, attempts and cooldowns survive interruption/resume. A retry
is never counted as a second final prediction. Successful terminal records are
not resubmitted. Known terminal failures also remain recorded and are skipped
on an explicit resume; authentication/billing errors pause for review rather
than being retried automatically. Study-specific terminal errors can continue,
with repeated errors pausing the run. No failure is treated as an exclusion.

Timeouts, unfinished tasks, malformed task envelopes and server errors have
unknown outcomes: they retain a pending marker and block automatic resume.
Inspect any task ID, private receipts and the provider dashboard. There is no
automatic pending deletion or repost. A per-run exclusive lock blocks concurrent
writers. A killed process can leave a stale lock, and a partial/corrupt JSONL
file also requires investigation. These are conservative local safeguards, not
a provider-side exactly-once guarantee.

Resume the same experiment/run IDs and inputs. Changes to ordered states,
source bytes, criteria, models, backend, source code or locked environment block
resume. Changed configuration files after plan review require rebuilding the
plan. Execution sessions record actual settings/checksums on explicit resumes;
attempt budgets are not reset. Existing completed pilot artifacts are retained,
but adding full-run source changes code hashes, so their old live resume guards
must not be bypassed or silently relabelled. Read historical results separately
when needed.

### Validation, provenance and evaluation export

Section 18 loads saved results without running or resuming inference. It compares
expected/recorded counts, successes, failed/unknown/textless/missing records,
duplicate/unexpected IDs, probability shape/range/finiteness, missing task IDs
and model metadata, provider-name differences and incompatible provenance.
Known returned-name variants are counted separately from unexpected identities.
Raw malformed responses remain available in private receipts. A complete handled
cohort with failures or textless studies is `completed_with_exceptions`, not an
assertion of complete probability coverage.

Section 19 reads stored experiment/run IDs, requested and returned model names,
explicitly reported model revisions, input/question/config/source/lockfile hashes,
Git state, environment versions, UTC times, attempts, errors, available token
usage and completion state. Unreported revisions remain null. A reported revision
is labelled provider-reported and unverified; a returned name or deployment
version does not establish immutable model weights. No headers containing API
credentials are exported.

Section 20 joins by the stable `study_id`. The prepared CSV mapping is explicit
(`0 -> false`, `1 -> true`); IDs and ITEM_ID are read as strings, preserving leading
zeros. Duplicate/unexpected reference IDs and unmapped labels block the join.
Missing labels stay nullable. The current reference has all 7,008 IDs, 1,312
missing ITEM_ID values and one shared ITEM_ID group; ITEM_ID is never the join key.

The table has exactly two rows per source study, unique by study/model, including
failed, unknown, textless and unrecorded outcomes. It uses a validated many-to-one
reference join and checks that no row was lost or duplicated. Malformed successful
predictions or incompatible provenance block export. No probability cutoff or
performance metric is calculated here; notebook 04 remains the analysis stage.

With `EXPORT_FULL_RESULTS = True` and both YAML write permissions, a uniquely
named export bundle records CSV, reference and prediction-file checksums, cohort
and task hashes, row counts, label mapping and missingness/validation. Partial
coverage is explicitly recorded. Existing export bundles are not overwritten.
Only aggregate output is displayed by default; row-level tables and all executed
notebooks remain private. Git exclusions and commit checks remain in effect.
Existing saved pilot outputs are preserved locally, so the source notebook must
still have outputs cleared before it can pass the commit guard.

All new tests use invented studies, temporary reference CSVs and mock HTTP;
fresh-kernel tests execute only the appended sections in dry and mocked-live
modes, as well as the preserved pilot scenarios. No new packages, GPU runtime,
model weights, API calls or full inference are installed/executed during setup.

## Read-only evaluation of completed archives

Notebook 04 now selects full saved runs and prepared human CSV references, rather
than the shared synthetic demonstration profile. Current full coverage is 7,008
reference-labelled studies, 6,997 probability outputs per model and eleven
textless records with no inferred decision. CSV labels join by string study ID.

`load_saved_full_evaluation` verifies the frozen scientific inputs and output
integrity, then uses the archived run's own fingerprint/code/environment for
read-only reporting. It does not call `plan_run` to construct a fresh live
fingerprint, does not create a provider client, does not resume a run and does
not alter old metadata. The live runner's stricter resume guards remain unchanged.

The analysis YAML has an unset threshold, a disabled partial-metrics option and
exports disabled by default. No arbitrary cutoff is inferred. Once a threshold
is explicitly specified, probability >= threshold means retain; failures and
unavailable outputs stay missing. Complete-cohort headline metrics remain
withheld when labelled coverage is incomplete. An explicit partial-metrics option
reports conditional performance with coverage shown alongside it.

Optional analysis bundles preserve reference/data/task and prediction checksums,
original inference source/environment identities and the current analysis
implementation/settings. The real-data evaluation and synthetic teaching path
are tested in fresh kernels with inference entry points and HTTP blocked.

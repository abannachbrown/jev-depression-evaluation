# Reproducibility

Use `uv sync --locked` and retain `uv.lock` in Git.

For every evaluation, save a private manifest under the run's results
directory containing:

- A unique run ID and UTC start/end timestamps.
- Git commit and whether the working tree had uncommitted changes.
- Python and uv versions, platform, and the `uv.lock` checksum.
- Resolved configuration with credentials removed.
- Exact task template and response-schema contents or their checksums.
- Dataset checksum, exact record count, and processing order.
- Requested model IDs and any model revisions or fingerprints returned.
- Generation settings and whether an API seed was supported and sent.
- Request IDs, attempts, failures, latency, and available usage totals.
- Scoring definitions and the handling of invalid or missing responses.

Keep predictions, raw responses, and manifests under `results/`. They may
contain evaluation data and must remain outside Git. Do not copy credentials
or authorization headers into run manifests or logs.

A local seed and temperature zero do not guarantee identical hosted
responses. Record provider model versions when available and document
any provider-side limitations. Avoid mutable model aliases for final runs
where immutable versions are available.

Resume only when the dataset, model identity, task template, generation settings,
and evaluation configuration match the original run. Preserve completed
records and record terminal failures explicitly. Distinguish retryable failures
from successful requests to avoid unnecessary duplicate billing.

Hosted and local inference must expose a shared request/response contract.
Import local runtime dependencies only inside the local backend.

For local runs, additionally record weight and tokenizer revisions,
runtime versions, hardware, dtype, and quantization settings.

## Notebook execution and run records

The configuration loader, state validator, demo backend, batch runner, private
checkpoint writer, and binary evaluation are implemented. The hosted transport
requires an explicit adapter. The separate Inference.sh connection supports one synthetic state/question.
Notebook 03 additionally permits an explicitly enabled ten-state pilot with
mandatory persistence, a frozen shared sample and separate model run directories;
unconfigured Inference.sh batches remain blocked. See `docs/inference_sh.md`.
Local inference is unimplemented, disabled, and disallows weight downloads.

Open notebooks using the existing uv environment. Record a reviewed Git revision
for final research runs. The manifest also hashes package source files because
an uncommitted source change can otherwise share the same Git revision.

The ordered dataset fingerprint hashes normalized citation states, not raw file
bytes; differently formatted files with the same ordered states have the same
fingerprint. The task fingerprint hashes the selected NOUL question definitions.
Changing states/order, model settings/IDs, endpoint, task, generation settings,
Python version, source files, response-schema checksum, or lockfile prevents
resume. Directory permissions and request limits can change without changing
the scientific-input fingerprint; the attempt counter persists across resumes.

Notebook settings are in `configs/notebooks.yaml`. Paid calls and writes each
need two explicit permissions: YAML and the execution/export cell. The default
demo completes in memory. Writing creates a private run directory containing
`manifest.json`, `predictions.jsonl`, `responses.jsonl`, and `failures.jsonl`.
Raw-response recording follows `evaluation.output.save_raw_responses`.

The hosted adapter must supply requested/returned model identities, safe request
IDs, token usage, and raw responses when the provider exposes them. The transport
does not persist authorization headers or include raw provider errors in failure
messages. Monetary cost is unavailable without a pricing contract; demo latency
is not model timing evidence. The local sampling seed is not implicitly sent as
an unsupported API parameter.

## Resume and interruptions

Use a unique run ID. To resume, select the same ID, inputs, and settings and enable
writes. Successful and terminal failed records are not automatically resent.
The run fingerprint must match. Do not run concurrent writers for a run ID.

Checkpoints append only new JSONL records and replace the manifest last, avoiding
rewriting the entire batch after every request. Each write checkpoint records a
pending request before sending it. A process
interrupted while that request is pending may have already incurred charges;
automatic resume is blocked until the outcome is investigated. Connection setup
failures and configured retryable HTTP statuses can be retried within the
attempt budget. Other transport failures with an uncertain outcome are terminal.
Provider Retry-After delays take precedence over the exponential backoff cap.

## Evaluations and sharing

Reference labels join by citation ID and predictions join by citation/model ID.
Duplicate or unexpected IDs are rejected. Missing, failed, and invalid responses
remain visible; incomplete coverage suppresses headline metrics by default.
Undefined metric denominators produce unavailable values. Demo and real
predictions cannot be mixed. Handcrafted examples have no performance meaning.

Committed notebook templates must have no saved outputs, execution counts,
attachments, or widget/execution metadata. Store executed copies and all exports
under ignored `results/`. Install the pre-commit hook in each clone. The hook
checks staged blobs and CI checks committed blobs, including likely credentials.
Neither a hook nor a scanner can recognize all sensitive text pasted into source.
Notebook tests execute only synthetic demos in fresh kernels with HTTP disabled.

The supplied request templates are preserved under `prompts/drafts/`.
The original hybrid draft is incomplete JSON. The separately supplied corrected
hybrid is complete but not selected. Main evaluation configurations leave task
selection unset; the Inference.sh connection selects one binary draft question.

## Inference.sh pilot records

Notebook 03 explicitly loads `configs/workflows/pilot_jev_9b.yaml` and
`configs/workflows/pilot_jev_27b.yaml`, with separate execution cells and flags
for each model. Both profiles reuse a
private frozen sample manifest and the same version-controlled binary question.
The sample manifest records the prepared JSONL byte checksum, any conversion
manifest/workbook checksum, ordered sample IDs, seed, validation and criteria
fingerprint. Records without citation text are reported as unavailable for
inference and remain in the source/reference data; other validation errors block
sampling. No human decision or ITEM_ID enters the three-field model state.

The pilot requires explicit paid-call/write flags, one model, ten frozen states,
one NOUL, at most ten attempts, sequential pacing and raw preservation. Each
model has its own run directory. Rows retain probabilities and available usage;
receipts retain exact response bytes, task IDs, request hashes and UTC times.
A pending marker is cleared only after receipt and row persistence. Exclusive
run locks prevent concurrent writers. Ambiguous outcomes stop and block resume;
terminal failures and successes are skipped on explicit resume. Source/lockfile
or input changes prevent resume. Never select a fresh run ID to bypass an
unresolved request. A stale lock or partially written file requires review.

The shared YAML's generation parameters are not sent to Inference.sh. Neither
probability thresholds nor monetary prices are inferred. Inspect actual provider
charges, rate limits and result validity before scaling or deriving binary
performance metrics. All executed copies and artifacts remain outside Git.

Notebook 03 compares probabilities by original study ID in the frozen order,
retaining failed, unknown and missing responses. Its methods draft derives
protocol and observed counts from actual plans/results and identifies incomplete
execution rather than claiming that planned requests were performed. Completion
is described only for two ten-success runs with no pending state. Returned model
identifiers and Git/lockfile/input fingerprints are shown as available; no
immutable hosted weight revision or deterministic output is inferred. Both
model flags are currently enabled at the user's request; all automated execution
uses synthetic states and mock transports.

## Full evaluation provenance and recovery

Notebook 03 appends full-evaluation Sections 12–20 after its preserved 22 pilot
cells. Dedicated full YAMLs leave demo/pilot/connection/model settings unchanged.
Dataset counts are read dynamically. A shared immutable cohort manifest records
all ordered IDs/states, source/conversion/workbook hashes, the exact existing
eligibility task and textless-record audit. Every source study remains in the
cohort and final study/model table. Textless records receive explicit null
probabilities and no API request; no human screening decision is modified or
passed to a model.

The shared serial runner uses a separately enabled full policy, compact in-memory
records, exact response files, per-attempt receipts and append-only checkpoints.
A pending marker is written before submission and cleared only after durable
result storage. Only 429 rejections receive bounded automatic retries, respecting
backoff/Retry-After. Unknown outcomes block automatic resume. Counters and retry
cooldowns survive interruptions, and terminal successes/failures are not reposted
on ordinary resume. Input/code/lockfile incompatibility and exclusive run locks
prevent unsafe reuse. Each resume session records its actual settings/checksums.

Run manifests retain requested/returned models, provider-reported revisions or
null, source/question/config/code/environment identities, UTC times, token usage,
attempts, errors and status. Revisions are unverified unless established outside
this workflow; model names are not immutable weights. Large cohort ID/audit
lists are stored once in the shared manifest, not rewritten in every checkpoint.

A separate CSV adapter preserves string study/ITEM_ID values and explicitly
maps 0/1 human decisions to nullable Booleans. A many-to-one study-ID join creates
exactly two rows per source study, including all failed/missing/unknown/textless
results. Duplicates, invalid probabilities/metadata or incompatible provenance
block export. Export bundles include source prediction/reference/output hashes,
validation and missingness, and never silently overwrite an earlier bundle.
Threshold choice and performance metrics remain in notebook 04.

Implementation preserves existing pilot artifacts and resume fingerprints.
New source changes therefore cannot silently resume or relabel an old-code pilot
run. The full workflow defaults to dry mode, with paid calls and writes disabled
in its YAML and both full execution flags false. Do not use Run All: earlier
pilot flags remain enabled as requested. No live calls are made during automated
tests; kernel execution uses synthetic data and mock transports exclusively.

## Read-only full-run analysis

Notebook 04's default analysis configuration is
`configs/evaluations/analysis_full.yaml`; the general demonstration configuration
is unchanged. It reads the complete prepared states, separate reference CSV and
both archived full runs. Source byte/order and task checksums, requested model
identities, frozen cohort and prediction integrity must match. Human reference
labels remain separate from all model inputs.

Archived inference metadata retains the original fingerprint, source hashes and
environment. Analysis changes do not force inference reruns or rewrite history.
API execution/resume continues to rebuild and check its own live fingerprint.
The analysis export records its own settings/environment/code and source
prediction/reference checksums separately.

The current analysis threshold is explicitly unset at the user's request. Real
reference counts and probability coverage are displayed without assigning
Boolean decisions. Thresholded metrics require a documented threshold, use the
same cutoff for both models and preserve failed/unknown/textless outcomes. Full
labelled denominators are retained; incomplete coverage suppresses headline
metrics unless conditional valid-pair analysis is explicitly enabled and labelled.

The two-study teaching fixture remains an explicit demo option. It is no longer
the default input to notebook 04. All evaluation loading and tests are read-only
with respect to saved inference artifacts and require no provider credentials.

## Recall-target exploration

`select_recall_thresholds`, `threshold_readouts` and `threshold_tradeoff_curve`
operate read-only on saved full-evaluation rows. The declared target and recall
basis are retained in candidate tables. Selection uses exact order statistics,
inclusive thresholds and ties, with an explicit unattainable status when too
few scored reference-positive records exist. Positive labels with unavailable
scores remain in the full-labelled denominator. Scored-pair confusion matrices
never silently assign a model decision to these records.

Notebook 04 distinguishes its configured analysis (threshold still unset) from
post-hoc threshold exploration. The exploratory plots/readouts state that
selection and measurement use the same human labels. Candidate recall is not a
guarantee for unseen studies, and conditional/manual-retention scenarios must
not be reported as model-only or independent validation. The exact source and
reference hashes and raw model outputs remain unchanged. Human review routing
for unavailable records is a separate declared workflow choice.

Notebook 04 Section 10 compares scored-pair specificity, PPV (precision), F2
(`5 TP / (5 TP + 4 FN + FP)`) and WSS@95 at each displayed cutoff.
`threshold_readouts` reports `wss95_scored = (TN + FN) / N_scored_labelled - 0.05`
only when scored recall reaches 95%. `wss95_cohort_conservative` instead uses
`automatic_exclusions / N_all - 0.05`, requires complete reference labels and
full-labelled recall of at least 95%, and keeps every unscored record in the
manual-review queue. These use observed workload at inclusive cutoffs, without
interpolation through ties. Unattainable or undefined metrics remain unavailable;
negative WSS values are not clipped. Manual routing is not credited as model
recall, and none of these calculations alters inference artifacts or settings.

## Interim summary and planned validation

Notebook 05 reconstructs the development-cohort report using the existing
read-only archive loader and screening/reporting functions. Its own strict YAML
contains only the analysis-profile path, exploratory target/cutoffs, planned
validation size/status and report export settings. It does not modify the
official analysis threshold or inference configurations. The 2,000-study unseen
validation is metadata only and remains planned until after colleague review.

The report checks the conversion-manifest identity and, when recorded, the
reference CSV checksum before reporting aggregate preparation/audit findings.
It preserves original inference provenance separately from current report code,
configuration and environment identities. Paired error counts use only jointly
scored human-retain studies and report unavailable pairs separately.

Report exports require independent YAML/cell permission and use a unique
directory under ignored results. Aggregate HTML/CSV, PNG/PDF figures, methods
text and a checksum manifest contain no citation text or row-level identifiers.
The source notebook remains output-free. Confidence intervals, independent
validation performance and verification of immutable hosted weights are not
claimed by this interim summary. Review should establish the reference-standard
procedures, discrepancy handling, missingness policy and locked validation
protocol before unseen-study outcomes are examined.

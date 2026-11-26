# Methodological decision log

## 24 November 2026 — screening scope and reviewed wording v1

**Status:** agreed with the project owner during question review; awaiting
incorporation into the revised protocol and colleague review before validation.

The evaluation protocol's Sections 3.2–3.2.1 distinguish binary overall screening
from hybrid overall screening and criterion-combination rules. They require
identical overall question wording and conservative treatment of incomplete
title/abstract reporting. The technical plan requires prompt versioning and
separate storage of raw probabilities, derived rules and review annotations.

The project owner confirmed:

1. Ambiguous antidepressant-only behavioural studies should be retained, but
   antidepressant administration alone must not establish a depression model.
2. Authors' research framing must be considered together with the investigated
   phenotype. A study framed as bipolar disorder should be retained when a
   depression-related component is investigated; another psychiatric condition's
   name must not automatically exclude it.
3. Such a study may carry the separate descriptive flag
   `depression_with_other_psychiatric_condition`. Unclear disease framing may be
   annotated separately. These flags do not determine exclusion.

**Rationale:** preserve a potentially eligible depression component during
title-and-abstract screening and distinguish a supporting phenotype from a drug
name, test name, broad disorder label or unrelated meaning of "depression".

**Protocol implication:** the original review's named psychiatric-disorder
exclusions must not be silently carried over as blanket exclusions. The revised
evaluation protocol must explicitly describe the agreed depression-component
interpretation. This decision is a scope clarification/change, not a claim that
the historical protocol or historical screening decisions already used it.

**Implementation record:** question-only templates are stored under
`prompts/reviewed/v1/`. The binary and hybrid overall definitions match exactly.
The hybrid retains seven questions; only its first four domains are primary
combination inputs. Comparator/outcome assessments remain secondary. A private,
study-ID-linked annotation format is documented beside the templates.

**Effect on previous work:** original drafts, archived runs, human reference
labels and model outputs are unchanged. Existing workflows are not switched to
the new wording. Historical scores cannot be relabelled as predictions from v1.
Future execution must record the selected template, its content fingerprint and
new run identities. This decision does not authorise or perform a new API run,
settle the R1 adjustment-floor ambiguity, select thresholds, or use the planned
2,000-study validation dataset.

## 24 November 2026 — hybrid integration and R1 status

The project owner requested adapting notebooks 00–05 for hybrid execution and
explicitly selected **keep R1 disabled until the protocol is clarified**.
The central screening plan records this status. No interpretation of the
ambiguous 0.5 threshold floor has been selected.

Separate v1 binary/hybrid pilot/full profiles use the reviewed templates, shared
source/seed, explicit primary questions and distinct run/cohort identities.
Seven-question requests are an explicit backend/profile opt-in; the original
single-question connection retains its limits. Live flags and new permissions
start off. No inference was authorised or performed during implementation.

Read-only analyses distinguish hybrid direct, R2 minimum and R3 mean; raw scores
remain unchanged. Pooled target cutoffs are exploratory. Formal R2/R3 tuning
requires the original 1,251-record tuning subset and reconciliation of the cohort
discrepancy. Final rule selection and a validation lock are not implied by the
existence of candidate tables. Legacy illustrative cutoffs and results are not
silently transferred to the new wording.

## 24 November 2026 — enable the hybrid ten-study pilot

The project owner explicitly selected the ten-study hybrid pilot for both
models, with no repeat of binary screening. Notebook 03's two pilot flags and
their hybrid YAML paid/write permissions are enabled. Every model execution
cell now checks that its actual plan is hybrid before submitting; a binary or
stale single-question plan is rejected. Each model remains capped at ten states
and ten attempts, with seven NOULs per study. Full-cohort flags, permissions and
dry-run settings remain off/off/on respectively. These edits enable execution
by the user; no live request was made during configuration.

## 24 November 2026 — enable full-cohort hybrid inference

After reporting the pilot had run, the project owner requested full execution.
Both saved hybrid pilot manifests show completed status. Notebook 03 now enables
both full-model flags and disables its pilot flags. Both full hybrid YAMLs allow
paid calls and mandatory saving with dry-run disabled. The seven-question task,
separate model output paths, serial rate limit, request/attempt budgets, retry
policy and strict resume checks are unchanged. Binary execution remains blocked;
R1 remains disabled. This enables user execution of Sections 16 and 17 after
reviewing Sections 12–15; no live API request was made during configuration.

## 25 November 2026 — four-arm archive comparison

The project owner requested showing binary and hybrid NOUL for both JEV models
side by side in notebooks 03–05, preserving existing paid responses. The new
comparison YAML explicitly selects the original binary archives and reviewed
hybrid-v1 archives, independently of the setup default approach. No existing
run is relabelled, resubmitted or overwritten. Notebook 03's four execution
flags are off; its comparison cells reuse any existing archive, including a
partial one, rather than automatically resuming it.

The archived binary and hybrid overall eligibility definitions differ. Their
results are therefore a descriptive comparison of saved experiments, not an
isolated question-design effect. Cohort order, normalized input content and
human reference data must agree across all four arms. Coverage and original
provenance are shown per arm. Ranking metrics additionally use an identical
all-four-scored subset to expose denominator effects. Main performance uses
overall answers; R2 minimum and R3 mean remain secondary derived scores. No
final threshold, replacement tuning split, R1 interpretation or independent
validation claim is introduced.

The earlier notebooks are privately backed up under results/notebooks. Source
notebooks are output-free. No API request was made during these changes; no
source workbook, input, reference label, recovery audit or inference result
was changed.

## 26 November 2026 — shared setup and preparation for all four arms

At the owner's request, notebooks 00–02 now make both model deployments and both
NOUL designs explicit. Setup validates four configurations without opening
private citation data. Preparation checks each arm's selected inputs against one
ordered citation cohort and one mapped human reference, retaining all studies
and reporting missingness. It does not reconvert the workbook, duplicate the
dataset, freeze a new pilot sample or replace original tuning membership.

The connection notebook keeps the original individual live-test limits and
save-only recovery, defaults its binary walkthrough to a separate mock profile,
and adds four labelled mock request/response examples plus a saved-metadata
inventory. These examples reject live/paid/write-enabled profiles and do not
measure model performance. Earlier executed notebooks are privately backed up;
source outputs are cleared. Original connection/full/pilot artifacts and real
data are preserved. No provider request was made and no package was installed.

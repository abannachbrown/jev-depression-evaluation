# Screening wording v1 — owner review, 24 November 2026

These question-only JSON templates record the project owner's screening-policy
clarifications. They are not a locked validation release, a completed colleague
review, or evidence of new inference. The v1 screening plan now selects these
templates explicitly; archived legacy results retain their original definitions.

| File | Questions | Overall question ID |
| --- | --- | --- |
| `binary_noul.json` | One overall screening question | `retain_for_fulltext` |
| `hybrid_noul.json` | Six domain questions plus the overall question | `overall_eligibility` |

The overall question definitions are identical, including their instructions
and true/false criteria. Their different IDs are an explicit mapping, not a
wording difference. Models and citation states are supplied separately by the
workflow; these templates contain no mutable model alias or example state.

## Agreed interpretation

- Retain ambiguous antidepressant-only behavioural studies. Antidepressant
  administration alone is not evidence of a depression model.
- Consider the authors' aims, manipulation and phenotype interpretation.
  Neuropsychiatric framing alone is neither sufficient nor disqualifying.
- Retain a depression-related component alongside another psychiatric condition,
  including a study framed as bipolar disorder. Clearly unrelated uses of
  "depression", such as respiratory depression, do not establish that component.
- Do not apply a blanket exclusion based on another psychiatric disorder's name.
  This clarification changes the interpretation of the original review's named
  disorder exclusions; it must be incorporated into the revised evaluation
  protocol before validation is frozen.
- Missing or ambiguous reporting favours retention. Entirely textless records
  remain unscored and enter manual review; no probability is imputed.
- Only `primary_research`, `animal_study`, `in_vivo_experiment` and
  `depression_model` contribute to primary criterion-combination rules.
  `control_group` and `eligible_outcomes` remain secondary assessments and cannot
  independently cause primary exclusion.

The v1 backend/runner supports explicit seven-question requests and saves all
answers. Direct, R2-minimum and R3-mean analyses are separate; the latter two
are derived scores, not calibrated joint probabilities. R1 remains disabled by
the project owner's decision pending clarification of its threshold floor.
Formal tuning requires the original reconciled development/tuning membership.
The planned independent-validation workflow remains unused. New wording receives
new task/run identities; it cannot resume a legacy experiment.

## Separate manual review annotation

Use a private review file linked by the exact original `study_id`, for example
`data/private/review_annotations.csv`. This is an annotation convention, not an
automated annotator or an eighth NOUL question. It does not alter the reference
label, model probability or primary retention decision and is never a model-input
field. Private evidence notes must stay out of Git and aggregate reports.

Suggested CSV header:

```csv
study_id,prompt_version,review_flag,condition_as_reported,evidence_source,evidence_note,reviewer,reviewed_at_utc
```

| Field | Meaning |
| --- | --- |
| `study_id` | Original string ID, preserving leading zeros |
| `prompt_version` | Version whose wording was reviewed, such as `screening-v1`; do not retrospectively relabel an old run |
| `review_flag` | `depression_with_other_psychiatric_condition` or `unclear_disease_framing` |
| `condition_as_reported` | Author-reported condition; leave empty when unclear, rather than infer a diagnosis |
| `evidence_source` | `title`, `abstract` or `title_and_abstract` |
| `evidence_note` | Supporting quotation/description from the supplied citation |
| `reviewer` | Person making the annotation |
| `reviewed_at_utc` | UTC timestamp of the annotation |

Assign the specific condition flag only when the citation supports both the
depression component and the other condition. Otherwise use the uncertainty flag
where appropriate. An annotation is a finding for review, not a new exclusion
rule. No real annotation records are included here.

See [the decision log](../../../docs/methodological_decisions.md).

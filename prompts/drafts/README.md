# Draft NOUL request templates

These supplied files are preserved byte-for-byte, with normalized filenames:

- `binary_noul.json` came from `Binary_Noul_Instructions.json`. It is valid JSON
  containing a complete request template with one `retain_for_fulltext` NOUL
  question and true/false criteria.
- `hybrid_noul.json` came from `hybrid_Noul_instructions.json`. It is incomplete
  JSON: it ends after the `depression_model` instruction, without that question's
  criteria or the closing structure. Complete it before parsing or using it.

Both files include a `model`, an example `state`, and `questions`; they are full
draft request templates rather than standalone prompt strings. Their placeholder
state is synthetic. Never replace it with real evaluation records in Git.

The drafts retain the supplied `jev-latest` model value. This mutable alias is
not the configured identity of either benchmark model. A future adapter must
select the model from the model configuration and insert each dataset state.

Existing binary evaluation profiles select `binary_noul.json`. The original
truncated hybrid draft is not selected; no missing hybrid criteria or aggregation
rules have been inferred from that incomplete file.

## Additional corrected hybrid draft

`hybrid_noul_corrected.json` is a separately supplied, complete JSON draft with
seven NOUL questions, including `overall_eligibility`. It is preserved unchanged.
The original truncated draft is retained for provenance. The corrected draft is
not selected by the binary notebook workflow; a reviewed choice of primary
question, reference labels, and any hybrid aggregation rules is still required.

## Owner-reviewed wording v1

The agreed 24 November 2026 screening clarifications are captured separately in
[`../reviewed/v1/`](../reviewed/v1/README.md). Those templates have identical
binary/hybrid overall questions and document separate manual review flags.
The new screening-v1 profiles select them explicitly, with hybrid as the central
default. No new model calls are implied by that configuration. Legacy workflows
and saved runs retain the supplied binary draft. The drafts here remain unchanged.

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

The evaluation YAML files deliberately leave `task.prompt_file` unset. Before
wiring these drafts into a run, finalize their structure, confirm the provider's
NOUL contract, and define response parsing and scoring. No missing hybrid
criteria or aggregation rules have been inferred.

## Additional corrected hybrid draft

`hybrid_noul_corrected.json` is a separately supplied, complete JSON draft with
seven NOUL questions, including `overall_eligibility`. It is preserved unchanged.
The original truncated draft is retained for provenance. The corrected draft is
not selected by the binary notebook workflow; a reviewed choice of primary
question, reference labels, and any hybrid aggregation rules is still required.

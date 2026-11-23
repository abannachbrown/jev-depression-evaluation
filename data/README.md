# Data storage

`examples/states.jsonl` contains two invented pipeline fixtures using the
`study_id`, `title`, and `abstract` fields from the supplied draft state.
`examples/jev_citation_state.json` is the original supplied example state,
preserved unchanged. These files contain placeholders, not evaluation records
or reference labels.

Store real evaluation data under `private/` or outside this repository.
The default real-data location is `private/states/`, with one JSON state
per file. `JEV_DATA_DIR` can override that directory.

Only the explicitly named synthetic examples are allowed through `.gitignore`.
Do not copy real records, participant identifiers, or reference labels into them.

Before a final evaluation, freeze the input dataset and record its exact
record count and SHA-256 checksums in the private run manifest.

## Reference labels

The notebook workflow accepts a separate JSONL reference-label file:

```json
{"study_id":"synthetic_001","labels":{"retain_for_fulltext":true}}
```

The task question ID must match `configs/scoring/binary.yaml`. Values must be
JSON booleans, not strings or integers. IDs must be unique and present in the
selected dataset. A labelled subset is allowed and its size is reported.
The loader does not infer ground truth from titles, abstracts, or predictions.

`examples/reference_labels.jsonl` and `examples/synthetic_predictions.jsonl`
are handcrafted teaching fixtures. Their decisions are invented and have no
scientific meaning. They demonstrate complete and contrasting confusion counts;
they do not represent performance of the named models. Do not replace their
contents with real study records, annotations, or API responses.

Store real reference labels under `private/` or outside the repository, and
select them through `configs/notebooks.yaml`. Generated predictions always
belong under `results/`, never in the examples directory.

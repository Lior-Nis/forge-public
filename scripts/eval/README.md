# Evaluation commands

`release/manifest.yaml` is the source of truth for released checkpoints,
datasets, thresholds, and expected values.

## Public reproduction

Run the four external cohorts through the root entrypoint:

```bash
./reproduce.sh
```

That command builds the public datasets, runs `eval_comprehensive.py`, derives
the cohort-level metrics with `eval_external_cohorts.py`, and verifies them
against the manifest.

## General evaluation

| Script | Purpose |
|---|---|
| `eval_comprehensive.py` | Shared inference engine across datasets, contexts, and model phases. |
| `eval_external_cohorts.py` | AUROC, AP, and ICC(%TF) for the released external evaluation. |
| `compute_icc_thresholds.py` | Threshold-specific clinical ICC summaries. |
| `eval_pooled_ap.py`, `pooled_ap_from_logs.py` | Fold-level pooled-AP aggregation. |
| `eval_embedding_probe.py`, `eval_knn.py` | Representation-quality probes. |

The remaining scripts are focused analyses named after their output. Scripts
that depended on unpublished, author-local prediction vectors are not included
in the public repository.

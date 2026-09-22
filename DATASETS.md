# Dataset provenance and terms

FORGE code is MIT-licensed. That license does **not** relicense third-party
recordings, labels, or metadata. Dataset users remain responsible for the terms
and participant-data restrictions attached to each source dataset.

The exact revisions and machine-readable cohort roles used by the released
detector are recorded in [`release/manifest.yaml`](release/manifest.yaml).

| Cohort | Role in FORGE | Public location used by reproduction | Notes on provenance and use |
|---|---|---|---|
| DeFOG | Detector training and in-distribution reference | `Liornis/fog-dataset/kaggle_labeled/defog` | Originates from the Parkinson's Freezing of Gait Prediction data release. It is intentionally excluded from the external-cohort reproduction command. |
| tDCS-FOG | External, provoked-task evaluation | `Liornis/fog-dataset/kaggle_labeled/tdcsfog` | Resampled from 128 Hz to 100 Hz by the public pipeline. Evaluation includes all 71 released participants. |
| FogAtHome-provoking | External structured evaluation | `Liornis/fog-dataset/fogathome` | Released for this joint research package. The evaluator retains frames marked valid by the released annotations. |
| FogAtHome daily living | External naturalistic evaluation | `Liornis/fog-dataset/fogathome_dailyliving` | Released for this joint research package. The reported result is restricted to walking/standing using the public `activity.parquet` sidecar. |
| Stanford | External cross-device evaluation | [`stanfordnmbl/imu-fog-detection`](https://github.com/stanfordnmbl/imu-fog-detection) | Downloaded directly from the source repository at the commit pinned in the release manifest; FORGE does not redistribute it. |

The Hugging Face dataset card should be treated as the current distribution
notice. If a source dataset's terms conflict with this repository's
documentation, the source dataset's terms govern.

## Evaluation contract

The released external evaluation uses the frozen medium-context encoder, nine
BiGRU heads (three participant folds across seeds 42, 43, and 44), and threshold
0.35. Run `./reproduce.sh` for the complete evaluation or
`uv run python reproduce.py --verify-assets-only` to check public packaging
without running inference.

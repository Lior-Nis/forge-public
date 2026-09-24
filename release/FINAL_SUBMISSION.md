# Final submission release verification

Final tag: `npj-2026-final-submission-v2`.
Hugging Face model revision: `3997f782f7567db357ee5caf0b134729a292f835`.
Dataset and Stanford revisions remain pinned in `manifest.yaml`.
The prior `npj-2026-submission-v1` tag is preserved unchanged.

Verification performed for this release:

- Anonymous public downloads and SHA-256 verification of the shared MC encoder
  and all nine MC probe classification artifacts.
- Embedded metadata covers DeFOG folds 0, 1, 2 for each of seeds 42, 43, 44.
- All nine artifacts rebuild through the public model loader and contain a
  bidirectional GRU. Their 52 backbone tensors are identical across members and
  match the standalone pretrained MC encoder.
- Anonymous `reproduce.py --verify-assets-only` passes for all 36 model artifacts,
  the pinned public dataset directories and daily-living activity sidecar, and
  the pinned Stanford source revision.
- Release tests: 35 passed, one skipped (legacy checkpoint unavailable).
- The documented public workflow downloads safetensors and raw data, builds its
  evaluation inputs, runs all nine heads, and creates prediction caches locally.
  No private checkpoints, private paths, or pre-existing predictions are required.

The Hugging Face model card describes the nine-head frozen-encoder ensemble,
identifies all fold/seed members, and states research-only, not-clinically-validated
use. Historical tDCS AP and the existing public reproduction target are explicitly
distinguished. No weights, scientific result values, evaluation logic, or data
were changed. Full cohort inference was not rerun during this packaging audit.

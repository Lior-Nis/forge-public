# FORGE Release Runbook (manual, network/GPU steps)

The Python pieces are tested offline. These steps publish to HuggingFace and run GPU eval —
run them deliberately, one at a time. Nothing here deletes local checkpoints.

1. **Build the manifest and verify the safetensors release set** (offline, safe):
   `source .venv/bin/activate && python scripts/release/manifest.py`
   `source .venv/bin/activate && python scripts/release/export_checkpoints.py`
2. **Build the compact DailyLiving Activity sidecar** (offline, safe):
   `python scripts/release/build_dailyliving_activity.py --source ~/Datasets/fogathome_dailyliving/preprocesseddata --output ~/Datasets/fog-dataset/fogathome_dailyliving/activity.parquet`
3. **Dry-run uploads** (prints plan, no network writes):
   `python scripts/release/upload_weights_hf.py`
   `python scripts/release/enrich_dataset_hf.py`
4. **Execute uploads** (creates public repo + uploads — review the dry-run first):
   `python scripts/release/upload_weights_hf.py --execute`
   `python scripts/release/enrich_dataset_hf.py --execute`
5. **Verify on HF:** the weights repo lists 36 `.safetensors` artifacts and checksums for
   all 36; the dataset repo includes `fogathome_dailyliving/activity.parquet`.
6. **Pin the new HF revisions** in `scripts/release/manifest.py`, regenerate
   `release/manifest.yaml`, then run `./reproduce.sh` from a clean checkout. It must
   check 12 metrics across FogAtHome-provoking, tDCS-FOG, DailyLiving, and Stanford.

Do NOT run step 4 or 6 while a training job is using the GPU.

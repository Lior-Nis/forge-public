# FORGE Release Runbook (manual, network/GPU steps)

The Python pieces are tested offline. These steps publish to HuggingFace and run GPU eval —
run them deliberately, one at a time. Nothing here deletes local checkpoints.

1. **Build manifest + export weights-only files** (offline, safe):
   `source .venv/bin/activate && python scripts/release/manifest.py`
   `source .venv/bin/activate && python scripts/release/export_weights.py --source <dir of trained .ckpt>`
   (or `--from-local` to read each entry's `local` path)
2. **Dry-run uploads** (prints plan, no network writes):
   `python scripts/release/upload_weights_hf.py`
   `python scripts/release/enrich_dataset_hf.py`
3. **Execute uploads** (creates public repo + uploads — review the dry-run first):
   `python scripts/release/upload_weights_hf.py --execute`
   `python scripts/release/enrich_dataset_hf.py --execute`
   Add `--squash-history` only when superseded files must not stay retrievable from earlier
   revisions. It is irreversible, so back the repo up first.
4. **Verify on HF:** weights repo lists 30 `.safetensors` + manifest + README; dataset repo now has
   `fogathome_dailyliving/`.
5. **End-to-end acceptance (GPU):** on a clean checkout, run the `onboard-repo` skill and
   confirm the MC-probe smoke test reproduces seg AUC ≈ 0.908.

Do NOT run step 3 or 5 while a training job is using the GPU.

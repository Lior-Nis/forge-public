"""Generate HuggingFace model + dataset card markdown from the manifest."""
from __future__ import annotations

def _result(m: dict, rid: str) -> dict:
    """Look a result up by id -- never by list position (the order changes)."""
    for r in m["results"]:
        if r["id"] == rid:
            return r
    raise KeyError(f"no result with id {rid!r} in manifest")


def _fmt(r: dict) -> str:
    ci = r.get("ci")
    return f"{r['value']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def model_card(m: dict) -> str:
    icc = _result(m, "fogathome_icc_tf")
    ladder_rows = "\n".join(
        "| {} | {} | {} | {} |".format(
            label,
            _fmt(_result(m, f"{key}_auroc")),
            _fmt(_result(m, f"{key}_ap")),
            _fmt(_result(m, f"{key}_icc_tf")),
        )
        for key, label in (
            ("fogathome", "FogAtHome-provoking (12) — cross-study"),
            ("tdcs", "tDCS-FOG (71) — cross-protocol"),
            ("stanford", "Stanford (7) — site / device / med state"),
            ("dailyliving", "FogAtHome daily living (11) — naturalistic*"),
        ))
    enc_rows = "\n".join(
        f"| {c.upper()} | {e['window_frames']} | `{e['hf_path']}` | {e['params']:,} |"
        for c, e in m["encoders"].items())
    cls_rows = "\n".join(
        f"| `{c['hf_path']}` | {c['context']} | {c['phase']} | {c['fold']} |"
        for c in m["classification"])
    return f"""---
license: mit
tags: [freezing-of-gait, parkinsons, accelerometer, self-supervised, mae, time-series, time-series-classification]
pipeline_tag: other
---

# FORGE — FOG Representation via Generative Encoding

Self-supervised spectral-temporal encoders for **Freezing of Gait (FOG)** detection from a
single lower-back accelerometer. Pretrained by masked autoencoding on 11,724 h (~21M
windows) of unlabeled at-home recordings from 65 participants, then trained for FOG
detection on the 57-participant DeFOG cohort only. Evaluated with no target-cohort
training on four external cohorts: FogAtHome-provoking, tDCS-FOG, Stanford and
FogAtHome daily living.

**Headline:** the released MC probe ensemble reaches clinical-grade agreement with expert
video annotation on an independent cohort — **ICC(%TF) = {icc['value']:.3f} [{icc['ci'][0]:.3f}, {icc['ci'][1]:.3f}]**,
zero-shot, one IMU.

## External results (released detector)

Nine-head MC frozen-probe ensemble (3 participant folds x 3 seeds), evaluated with no
target-cohort training and the unchanged DeFOG operating point of **0.35**.

| Cohort (N) — shift | AUROC | AP | ICC(%TF) |
|---|---|---|---|
{ladder_rows}

\\* Daily living is scored inside a label-independent walking-and-standing domain
(58.18 h of 301.8 h, 2.92% FOG); it is gait-conditioned burden, **not** whole-recording
%TF. Stanford is negative evidence: discrimination survives the shift, the fixed
threshold does not (its oracle-rule threshold is 0.18). In-distribution reference:
window-level AP **0.730** on the DeFOG validation folds. Full definitions and confidence
intervals are in `manifest.yaml` under `results:`.

## Controlled comparison (a different model set)

The paper's headline effect is a separate, deliberately constrained experiment: two arms
differing **only** in encoder initialization, under matched downstream training. Its
numbers are lower than the released detector's on the same cohort because it is a
different model set — not a worse estimate of the same thing.

| Cohort | Metric | Self-supervised | Supervised from scratch | Difference [95% CI] |
|---|---|---|---|---|
| FogAtHome-provoking | AUROC | 0.861 | 0.752 | +0.109 [0.029, 0.182] |
| FogAtHome-provoking | AP | 0.784 | 0.592 | +0.192 [0.064, 0.338] |
| tDCS-FOG | AUROC | 0.869 | 0.657 | +0.212 [0.126, 0.280] |
| tDCS-FOG | AP | 0.811 | 0.501 | +0.310 [0.133, 0.397] |

Each result in `manifest.yaml` names its `model` set and the manuscript `table` it comes
from, so the two sets stay distinguishable.

## Released weights

### Pretrained FORGE encoders (the backbones)
| Context | Window (frames) | File | Params |
|---|---|---|---|
{enc_rows}

### Downstream classification weights (57-participant DeFOG, 3-fold participant-level CV)
| File | Context | Phase | Fold |
|---|---|---|---|
{cls_rows}

## What this release contains
All 27 classification heads are the **seed-42** runs. The manuscript's released detector
averages nine heads (3 folds x 3 seeds) over one shared frozen encoder, so
`classification/mc_probe_fold{0,1,2}.safetensors` rebuild the **seed-42 three-fold ensemble**.
That is the configuration this project's evaluation scripts run, and it lands within about
0.02 of the nine-head numbers tabled above.

## Usage
This release is **weights only**: each file is a `.safetensors` tensor set with small
string metadata (name, context, phase, fold, seed, and the `experiment` config that
rebuilds the model). No training configuration, optimizer state or local file path is
included, and loading executes no pickled code.

Rebuild a model from the companion repo, which composes the architecture from the
`experiment` config named in the file's metadata and in `manifest.yaml`:

```python
from utils.released_weights import load_released_model

model, config = load_released_model(
    "release/forge-fog/classification/mc_probe_fold0.safetensors",
    experiment="classification/spectral_patch_mae_mc_valid_defog_soft",
    overrides=["data/splits=kaggle_labeled/kfold_defog_fogcount_valid_mc0"],
)
```

Or read the tensors directly:

```python
from safetensors.torch import load_file
from safetensors import safe_open

state_dict = load_file("classification/mc_probe_fold0.safetensors")
with safe_open("classification/mc_probe_fold0.safetensors", framework="pt") as f:
    meta = f.metadata()   # name / kind / context / phase / fold / seed / experiment / splits
```

Each classification file already contains its encoder, so `encoders/*.safetensors` are
needed only to train new heads.

## Citation
{m['meta']['authors']}. *{m['meta']['paper']}*. Submitted to npj Digital Medicine, 2026.

Reproduce every paper number with the companion repo's `reproduce-evaluations` skill (see
`manifest.yaml`, shipped in this repo).
Code: [github.com/Lior-Nis/forge-public](https://github.com/Lior-Nis/forge-public).
Data: [{m['meta']['hf_dataset_repo']}](https://huggingface.co/datasets/{m['meta']['hf_dataset_repo']}).

License: **MIT**.
"""

def dataset_card(m: dict) -> str:
    rows = "\n".join(
        f"| {k} | {d['hf_subdir']} | {d['role']} | {d.get('prevalence','-')} |"
        for k, d in m["datasets"].items())
    return f"""---
license: other
license_name: source-study-data-use-terms
tags: [freezing-of-gait, parkinsons, accelerometer]
extra_gated_prompt: >-
  These are human-participant recordings from clinical studies, released for
  non-commercial research use. By requesting access you agree to use them for
  research only, to make no attempt to re-identify participants, not to
  redistribute them, and to comply with the data-use terms of each source study.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Intended research use: text
  I will not attempt to re-identify participants: checkbox
  I will not redistribute this data: checkbox
---

# FOG Dataset (FORGE)

Lower-back accelerometer (Axivity, 100 Hz, 3-axis) recordings for FOG research.

**Access is gated.** Requests are reviewed by the authors. The recordings come from
third-party clinical studies and stay subject to the data-use terms of each source
study, so this repository is not covered by the MIT license of the FORGE code and
weights.

| Split | Subdir | Role | Frame FOG prevalence |
|---|---|---|---|
{rows}

`fogathome_dailyliving` holds 301.8 h of naturalistic free-living recordings (11 patients,
3,428 recordings, 1.09% frame-level FOG overall; 2.92% inside the walking-and-standing
eligible domain). Used as the external naturalistic evaluation in FORGE.
"""

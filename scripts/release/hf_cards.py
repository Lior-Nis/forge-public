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
        f"| `{c['hf_path']}` | {c['context']} | {c['phase']} | {c['fold']} | {c['seed']} |"
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

**Headline:** the released nine-head frozen-encoder ensemble is evaluated against expert
video annotation on an independent cohort — **ICC(%TF) = {icc['value']:.3f} [{icc['ci'][0]:.3f}, {icc['ci'][1]:.3f}]**,
using one lower-back IMU and no target-cohort training.

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
window-level AP **0.730** on held-out DeFOG folds. Full definitions and confidence
intervals are in `manifest.yaml` under `results:`.

## Released weights

### Pretrained FORGE encoders (the backbones)
| Context | Window (frames) | File | Params |
|---|---|---|---|
{enc_rows}

### Downstream classification checkpoints (57-participant DeFOG, 3-fold participant-level CV)
| File | Context | Phase | Fold | Seed |
|---|---|---|---|---|
{cls_rows}

## What this release contains
The release includes the nine MC probe heads (3 folds x seeds 42, 43, and 44).
External evaluation averages all nine; the DeFOG reference remains seed 42.

## Usage
Classification artifacts are safetensors files whose metadata names the committed Hydra
experiment and split used to rebuild the model.

```python
from utils.released_weights import load_released_model
model, config = load_released_model(
    "release/forge-fog/classification/mc_probe_fold0.safetensors"
)
```

Reproduce the four external cohorts with the companion repo's `./reproduce.sh` command
(see `manifest.yaml`, shipped in this repo).
Code: [github.com/Lior-Nis/forge-public](https://github.com/Lior-Nis/forge-public).
Data: [{m['meta']['hf_dataset_repo']}](https://huggingface.co/datasets/{m['meta']['hf_dataset_repo']}).

## Intended use and limitations

FORGE is a research model for evaluating freezing-of-gait methods on compatible
lower-back accelerometer recordings. It is not clinically validated and is not a medical device and must not be
used to diagnose, monitor, or make treatment decisions for an individual. The
fixed threshold is cohort-sensitive: the Stanford result in particular shows
that discrimination can survive a device/site shift while calibrated burden does
not. Users should report cohort provenance, sampling/resampling, eligibility
filters, and the exact model revision.

License: **MIT**.
"""

def dataset_card(m: dict) -> str:
    rows = "\n".join(
        f"| {k} | {d.get('hf_subdir', d.get('source', '-'))} | {d['role']} | "
        f"{d.get('prevalence','-')} |"
        for k, d in m["datasets"].items())
    return f"""---
license: other
license_name: source-specific dataset terms
license_link: https://github.com/Lior-Nis/forge-public/blob/main/DATASETS.md
tags: [freezing-of-gait, parkinsons, accelerometer]
---

# FOG Dataset (FORGE)

Lower-back accelerometer (Axivity, 100 Hz, 3-axis) recordings for FOG research.

| Split | Subdir | Role | Frame FOG prevalence |
|---|---|---|---|
{rows}

`fogathome_dailyliving` holds 301.8 h of naturalistic free-living recordings (11 patients,
3,428 recordings, 1.09% frame-level FOG overall; 2.92% inside the walking-and-standing
eligible domain). Used as the external naturalistic evaluation in FORGE.

## Provenance and terms

The MIT license in the FORGE code repository does not relicense participant
recordings or third-party datasets. Each cohort retains its source terms and
participant-data restrictions. The exact public revision used by the paper is
recorded in the companion repository's `release/manifest.yaml`; its
`DATASETS.md` identifies the source and evaluation role of each cohort.
"""

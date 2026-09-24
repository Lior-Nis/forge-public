# Reproducing the FORGE evaluation results

This runs the released frozen-encoder detector on the four external cohorts:
FogAtHome-provoking, tDCS-FOG, FogAtHome daily living, and Stanford. It uses
the shared MC encoder, all nine released BiGRU heads, and the fixed threshold
0.35. DeFOG is deliberately not part of this external reproduction command.

---

## Windows (easiest — no commands)

1. **Download the code.** On the GitHub page click the green **Code** button →
   **Download ZIP**. Save it and **extract** the ZIP (right-click → Extract All).
2. Open the extracted folder and **double-click `reproduce.bat`**.
3. A black window opens and does everything automatically:
   - installs the environment manager (`uv`) the first time,
   - sets up Python and all dependencies (uses your NVIDIA GPU if you have one),
   - downloads the model and data from the internet,
   - runs the evaluation and prints the results.
4. **Leave the window open** until it says `Done.` It will pause at the end so
   you can read the results. Full results are saved in the **`logs`** folder.

**Needs:** an internet connection and about **30 GB of free disk space**.
The first run downloads several GB, so it can take a while — that's normal.
If Windows shows a blue "protected your PC" box, click **More info → Run anyway**.

> Have an NVIDIA GPU (e.g. RTX 3060)? It's used automatically. No GPU? It still
> works on the CPU, just slower.

---

## Mac / Linux

```sh
git clone https://github.com/Lior-Nis/forge-public.git
cd forge-public
./reproduce.sh
```

---

## What you should see

At the end the script checks all 12 external metrics. AUROC and AP pass within
0.03 of the reference; ICC passes within its published interval.

```
                         AUROC    AP      ICC(%TF)
FogAtHome-provoking      0.887    0.804    0.899
tDCS-FOG                 0.917    0.866    0.876
Stanford                 0.734    0.400   -0.119
FogAtHome daily living   0.803    0.105    0.656
```

The daily-living row is restricted to walking/standing Activity codes 1 and 4.
The tDCS row consistently uses all 71 participants and the same nine-head
assembly for all three metrics; the older AP value 0.812 came from the stale
fold-safe analysis and is not the released-detector target.

If you see **`CHECKS PASSED`**, you're done.
(If anything says `FAIL`, send the window's text back.) The full numbers are
saved in `logs/comprehensive_eval.csv` and `logs/RESULTS_external.csv`.

The DeFOG number remains in `release/manifest.yaml` as a recorded
in-distribution reference, but it is not evaluated or used as a gate by
`reproduce.py`. Author-local analyses that depended on unpublished prediction
vectors are intentionally not part of the public repository.

---

## Options (optional)

To run only selected cohorts, pass their names, for example:

```sh
./reproduce.sh --datasets fogathome tdcsfog
```

The accepted names are `fogathome`, `tdcsfog`, `dailyliving`, and `stanford`.

FogAtHome and tDCS-FOG are downloaded from the pinned Hugging Face dataset
revision in `release/manifest.yaml`. Stanford is downloaded from the pinned
official `stanfordnmbl/imu-fog-detection` GitHub revision. Only `.safetensors`
model files are accepted.

If downloads are slow due to rate limits, signing in to a free Hugging Face
account first (`hf auth login`) speeds them up — but it works without one.

To verify the pinned public files and checksum coverage without downloading the
cohorts or running inference:

```sh
uv sync --frozen
uv run python reproduce.py --verify-assets-only
```

Every run writes `logs/reproduction_run.json` with the code revision, public
asset revisions, environment, command, runtime, and (after a completed
evaluation) the reproduced metrics.

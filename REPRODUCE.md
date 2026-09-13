# Reproducing the FORGE evaluation results

This reproduces the paper's headline numbers from the released model + data.
Everything is automated — you don't need to know Python.

> **Data access — do this first.** The evaluation data come from the Hugging Face dataset
> [`Liornis/fog-dataset`](https://huggingface.co/datasets/Liornis/fog-dataset), which is
> **gated**: open that page while signed in, accept the terms, and wait for the authors to
> approve the request. Then run `hf auth login` on this machine. Without approval the
> download step fails with an authentication error and the evaluation cannot proceed. The
> model weights ([`Liornis/forge-fog`](https://huggingface.co/Liornis/forge-fog)) are
> public and need no request.

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

**Needs:** an internet connection and about **20 GB of free disk space**.
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

At the end the script **checks the numbers for you** and prints a table like this
(illustrative — your "got" column will be close to, not identical to, these):

```
[PASS] FogAtHome frame AUROC         expected 0.887 +/-0.03        got 0.908
[PASS] FogAtHome frame AP            expected 0.804 +/-0.03        got 0.81
[PASS] FogAtHome ICC(%TF)            expected 0.899 [0.7, 0.97]    got 0.909
[PASS] DeFOG window AP (in-dist)     expected 0.730 +/-0.03        got 0.730

ALL CHECKS PASSED — results are in the manuscript's ballpark.
```

The expected column is the **manuscript's** released nine-head ensemble, scored on
annotator-verified frames; this script runs a seed-42 three-fold ensemble over the full
window grid. That is why "got" sits ~0.01-0.02 away rather than exactly on it — the check
is a ballpark, wide enough for that gap and narrow enough to catch a broken setup.

If you see **`ALL CHECKS PASSED`**, you're done.
(If anything says `FAIL`, send the window's text back.) The full numbers are
saved in `logs/comprehensive_eval.csv` and `logs/RESULTS_icc.md`.

### Scripts that cannot run from a clone

Eleven supplementary analysis scripts (`scripts/eval/kaggle_*.py`, `scripts/eval/eval_dl_comparison_table.py`,
`scripts/eval/build_kaggle_idmap.py`, `scripts/analysis/fig_kaggle_filter_robustness.py`,
`scripts/analysis/fig_ssl_collapse.py`) read prediction vectors from `research/paper_final/data/`, which is
gitignored private research material. They exit with an `[author-only]` message rather than a confusing
traceback. Set `FORGE_PRIVATE_DATA=/path/to/paper_final/data` if you have the tree.

**No headline result depends on them** — see the `verification` field on each entry in
`release/manifest.yaml`, which records for every reported number whether this repository can recompute it
(`one_click`), can with an extra pass (`scripted`), or cannot because the cohort is not in the public
release (`recorded`).

---

## Options (optional)

The default run reproduces the headline numbers (fastest). To run the full
paper table instead, add `--full`:

- Windows: open the folder, type `cmd` in the address bar, then `reproduce.bat --full`
- Mac/Linux: `./reproduce.sh --full`

If downloads are slow due to rate limits, signing in to a free Hugging Face
account first (`hf auth login`) speeds them up — but it works without one.

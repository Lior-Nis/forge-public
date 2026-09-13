#!/usr/bin/env bash
# run_embedding_analysis.sh
#
# Extract embeddings for the winning SSL model + supervised baseline, then
# regenerate all publication embedding figures (figs 3-6 in draft_v2.md).
#
# USAGE — point at whichever SSL method wins the fair comparison:
#
#   # MAE 2d_patch (current winner, ep7):
#   bash scripts/shell/run_embedding_analysis.sh \
#       --ssl-model-name  vit12_ep7_mae \
#       --probe-pattern   "checkpoints/classification/probe_vit12_ep7_fold{fold}/last.ckpt" \
#       --n-folds         5
#
#   # JEPA (if it wins):
#   bash scripts/shell/run_embedding_analysis.sh \
#       --ssl-model-name  vit12_ep7_jepa \
#       --probe-pattern   "checkpoints/classification/probe_jepa_fold{fold}/last.ckpt" \
#       --n-folds         3
#
#   # Single-checkpoint mode (supervised baseline — already the default):
#   # No change needed; the script always re-uses --sup-ckpt below.
#
# KEY ARGS:
#   --ssl-model-name   Short identifier used for the output directory and figure labels.
#   --probe-pattern    Glob pattern for the k-fold probe checkpoints.
#                      Must contain {fold} which is replaced with 0, 1, ... n-folds-1.
#   --sup-ckpt         Path to the supervised-from-scratch fold-0 checkpoint (for baseline).
#   --n-folds          Number of probe folds (default: 5).
#   --embed-dir        Root output dir for extracted embeddings (default: logs/embeddings).
#   --plots-dir        Output dir for publication figures (default: artifacts/plots/publication).
#   --probe-csv        Pooled-AP prediction CSV for probe model (fig4; optional).
#   --finetune-csv     Pooled-AP prediction CSV for finetuned model (fig4; optional).
#   --skip-extract     Skip embedding extraction, only re-run figures (if embeddings exist).
#   --figs             Space-separated list of figure numbers to generate (default: all).

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SSL_MODEL_NAME="vit12_ep7_mae"
PROBE_PATTERN="checkpoints/classification/probe_vit12_ep7_fold{fold}/last.ckpt"
SUP_CKPT="checkpoints/classification/supervised_scratch_fold0/last.ckpt"
N_FOLDS=5
EMBED_DIR="logs/embeddings"
PLOTS_DIR="artifacts/plots/publication"
PROBE_CSV=""
FINETUNE_CSV=""
SKIP_EXTRACT=0
FIGS=()

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --ssl-model-name)  SSL_MODEL_NAME="$2"; shift 2 ;;
        --probe-pattern)   PROBE_PATTERN="$2";  shift 2 ;;
        --sup-ckpt)        SUP_CKPT="$2";       shift 2 ;;
        --n-folds)         N_FOLDS="$2";        shift 2 ;;
        --embed-dir)       EMBED_DIR="$2";      shift 2 ;;
        --plots-dir)       PLOTS_DIR="$2";      shift 2 ;;
        --probe-csv)       PROBE_CSV="$2";      shift 2 ;;
        --finetune-csv)    FINETUNE_CSV="$2";   shift 2 ;;
        --skip-extract)    SKIP_EXTRACT=1;      shift ;;
        --figs)            shift; while [[ $# -gt 0 && ! $1 == --* ]]; do FIGS+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SSL_EMBED_DIR="${EMBED_DIR}/${SSL_MODEL_NAME}"
SUP_EMBED_DIR="${EMBED_DIR}/supervised_scratch"
SSL_LABEL="${SSL_MODEL_NAME//_/ }"  # underscores → spaces for figure labels

echo "=== Embedding analysis: ${SSL_MODEL_NAME} vs supervised ==="
echo "  SSL probe pattern : ${PROBE_PATTERN}"
echo "  SSL embed dir     : ${SSL_EMBED_DIR}"
echo "  Sup ckpt          : ${SUP_CKPT}"
echo "  Sup embed dir     : ${SUP_EMBED_DIR}"
echo "  N folds           : ${N_FOLDS}"

# ── 1. Extract SSL model embeddings ──────────────────────────────────────────
if [[ $SKIP_EXTRACT -eq 0 ]]; then
    echo ""
    echo "--- Extracting SSL embeddings ---"
    uv run python scripts/embed/extract_embeddings.py \
        --probe-pattern  "${PROBE_PATTERN}" \
        --model-name     "${SSL_MODEL_NAME}" \
        --n-folds        "${N_FOLDS}" \
        --output-dir     "${EMBED_DIR}" \
        --batch-size     64

    # ── 2. Extract supervised baseline embeddings ─────────────────────────────
    if [[ -f "${SUP_CKPT}" ]]; then
        echo ""
        echo "--- Extracting supervised baseline embeddings ---"
        # Derive defog patient list from the SSL embeddings we just produced
        uv run python scripts/embed/extract_embeddings.py \
            --single-ckpt    "${SUP_CKPT}" \
            --ref-metadata   "${SSL_EMBED_DIR}/metadata.csv" \
            --model-name     "supervised_scratch" \
            --output-dir     "${EMBED_DIR}" \
            --batch-size     64
    else
        echo "WARNING: supervised checkpoint not found at ${SUP_CKPT} — skipping baseline extraction."
        echo "         Pass --sup-ckpt to enable the comparison figures."
    fi
else
    echo "  (skipping extraction — using cached embeddings)"
fi

# ── 3. Generate publication figures ───────────────────────────────────────────
echo ""
echo "--- Generating publication figures ---"

PLOT_ARGS=(
    --primary-embed-dir  "${SSL_EMBED_DIR}"
    --baseline-embed-dir "${SUP_EMBED_DIR}"
    --primary-label      "${SSL_LABEL}"
    --baseline-label     "Supervised from scratch"
    --out-dir            "${PLOTS_DIR}"
)

[[ -n "${PROBE_CSV}" ]]   && PLOT_ARGS+=(--probe-csv   "${PROBE_CSV}")
[[ -n "${FINETUNE_CSV}" ]] && PLOT_ARGS+=(--finetune-csv "${FINETUNE_CSV}")
[[ ${#FIGS[@]} -gt 0 ]]   && PLOT_ARGS+=(--figs "${FIGS[@]}")

uv run python scripts/analysis/plot_publication.py "${PLOT_ARGS[@]}"

echo ""
echo "=== Done. Figures written to ${PLOTS_DIR} ==="

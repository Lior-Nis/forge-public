#!/bin/bash
# SSL pretraining-objective ablation under the all128 downstream protocol.
#
# Probes each SSL-pretrained SpectralPatchEncoder (frozen backbone + BiGRU head)
# on the 128-patient DeFOG 3-fold splits — identical config to the all128 FORGE
# probe of the SAME context, only the pretrained encoder (train.weights.load_from)
# changes. This makes Table 6 internally consistent: every method shares
# architecture (depth-4 spectral_patch_encoder), downstream head, data, splits,
# and eval. Only the pretraining objective differs.
#
# CONTEXT=lc: the original LC ablation. The SSL encoders were pretrained at LC,
# compared against the FORGE 2D-MAE LC probe (soft_probe_lc_all128). LC is FORGE's
# weakest context, so this comparison is fair but not flattering to 2D-MAE.
#
# CONTEXT=mc: the budget-matched MC ablation. Requires the MC re-pretrained
# encoders produced by scripts/shell/run_ssl_repretrain_mc.sh. Compared against
# the FORGE 2D-MAE MC probe (soft_probe_mc_all128) — the deployable headline
# context. Causal is excluded at MC (below random-init at LC).
#
# Usage: bash scripts/shell/run_ssl_ablation_all128.sh [method] [context]
#   method:  simclr|jepa|causal|mae1d|all  (default: all)
#   context: lc|mc                          (default: lc)

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD="${1:-all}"
CONTEXT="${2:-lc}"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"
case "$CONTEXT" in
  lc) EXP="classification/spectral_patch_mae_lc_valid_defog_soft"; BS=64 ;;
  mc) EXP="classification/spectral_patch_mae_mc_valid_defog_soft"; BS=256 ;;
  *)  echo "bad context: $CONTEXT" >&2; exit 1 ;;
esac

# method -> pretrained spectral_patch (depth-4) encoder checkpoint, per context.
# "random" is the control: no checkpoint, random-init backbone, still frozen —
# isolates what the GRU head + all128 labels extract from arbitrary fixed
# features (guards against the head/data masking a collapsed encoder).
declare -A CKPT_lc=(
  [simclr]="checkpoints/simclr/vulcan-bird-of-prey-205/last.ckpt"
  [jepa]="checkpoints/jepa/amber-microwave-203/last.ckpt"
  [causal]="checkpoints/mae/causal_mae_daily/last.ckpt"
  [mae1d]="checkpoints/mae/mae_1d_daily/last.ckpt"
  [random]="null"
)
# MC encoders from run_ssl_repretrain_mc.sh (budget-matched, 10-epoch budget).
# mae2d here is the budget-matched 10-epoch reference — NOT the 47-epoch
# deployment encoder. Causal omitted (below random-init at LC). If you re-pretrain
# with a different EPOCHS, update the mae2d budget tag below to match.
# NOTE: the ModelCheckpoint callback nests checkpoints under a run-name subdir
# (= the +train.logger.name set by run_ssl_repretrain_mc.sh, i.e.
# repretrain_mc_<method>_e<EPOCHS>). Paths below assume EPOCHS=5; update the
# e5 suffix if you re-pretrain with a different budget.
declare -A CKPT_mc=(
  [mae2d]="checkpoints/mae/mae2d_medcontext_budget5_daily/repretrain_mc_mae2d_e5/last.ckpt"
  [simclr]="checkpoints/simclr/simclr_medcontext_daily/repretrain_mc_simclr_e5/last.ckpt"
  [jepa]="checkpoints/jepa/jepa_medcontext_daily/repretrain_mc_jepa_e5/last.ckpt"
  [mae1d]="checkpoints/mae/mae1d_medcontext_daily/repretrain_mc_mae1d_e5/last.ckpt"
  [random]="null"
)
# select the active map for this context (bash nameref)
declare -n CKPT="CKPT_${CONTEXT}"

run_probe() {
    local method="$1" ckpt="${CKPT[$1]:-}"
    if [ -z "$ckpt" ]; then echo "[$method] no checkpoint defined for context=$CONTEXT — skip" && return; fi
    if [ "$ckpt" != "null" ] && [ ! -f "$ckpt" ]; then echo "MISSING ckpt for $method: $ckpt" >&2; exit 1; fi
    local load_override
    [ "$ckpt" = "null" ] && load_override="train.weights.load_from=null" || load_override="train.weights.load_from='${ckpt}'"
    echo "=== SSL PROBE: $method @ $CONTEXT  ($ckpt) ==="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/soft_probe_ssl_${method}_${CONTEXT}_all128_fold${fold}"
        [ -f "${OUT_DIR}/last.ckpt" ] && echo "[fold $fold] skip (exists)" && continue
        uv run python scripts/train/train_classification.py \
            experiment="${EXP}" \
            "${load_override}" \
            "train.weights.freeze_backbone=true" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "+train.logger.name=soft_probe_ssl_${method}_${CONTEXT}_all128_fold${fold}" \
            "train.logger.tags=[probe,frozen,ssl_ablation,${method},${CONTEXT},all128,fold${fold}]" \
            "data.dataloader.batch_size=${BS}"
    done
}

for m in mae2d simclr jepa causal mae1d random; do
    if [[ "$METHOD" == "$m" || "$METHOD" == "all" ]]; then
        run_probe "$m"
    fi
done

echo "=== SSL ablation probes done ==="

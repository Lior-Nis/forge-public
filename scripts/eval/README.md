# Evaluation scripts

**Start here:** `release/manifest.yaml` is the source of truth for which checkpoint + dataset
+ config backs each paper number, and the `reproduce-evaluations` skill drives the canonical
commands. This README signposts the scripts so an outside reader knows which is which.

## Canonical — the main paper results
| Script | Produces |
|---|---|
| `eval_comprehensive.py` | Primary engine: AP / AUC / NormAP across all datasets × contexts × models (incl. context ensembles). The `CKPT` / `ZARR` maps here are authoritative. `--threshold-source {test_youden,test_pr11,defog_val_pr11}`. |
| `compute_icc_thresholds.py` | Clinical ICC(%TF / #FOG / Duration) under both threshold protocols (Tables 1, S3; Figs 1, 11). |
| `eval_fogathome.py` | FogAtHome external eval entry point. |
| `eval_pooled_ap.py` / `pooled_ap_from_logs.py` | Pooled-AP aggregation across folds. |

## Daily-living head-to-head vs Kaggle winners (Fig 13 / §4.3)
| Script | Role |
|---|---|
| `kaggle_full_filters.py` | Main driver: FORGE vs 5 Kaggle winners under each gait filter. |
| `build_kaggle_idmap.py` | Recover the frame-index alignment to the winners' re-hashed sessions. |
| `kaggle_14_bootstrap.py`, `kaggle_normap_boot.py`, `kaggle_episode_sensitivity.py` | Significance / NormAP / episode-definition robustness bootstraps. |
| `dailyliving_gait_filter.py`, `eval_dailyliving_walkstand.py`, `eval_fogathome_walkstand.py` | Activity-conditioned (walk/stand) evaluation. |
| `kaggle_pred_vector_eval.py`, `kaggle_walking_bootstrap.py`, `kaggle_walking_fastboot.py` | Supporting / earlier-iteration bootstraps (kept for provenance; prefer `kaggle_full_filters.py`). |

## Label-efficiency & from-scratch curves (Fig S4 / §5)
| Script | Role |
|---|---|
| `eval_label_efficiency.py`, `eval_label_efficiency_patients.py`, `eval_label_efficiency_icc.py` | Label-budget curves (windows / patients / ICC). |
| `eval_scratch_lr_budgets.py`, `eval_scratch_lr_sweep.py` | Best-tuned supervised-from-scratch baselines per budget. |
| `bootstrap_significance.py` | Paired significance tests (probe vs supervised). |

## ICC across all datasets / supporting
| Script | Role |
|---|---|
| `eval_icc_all_datasets.py`, `eval_fog_icc_all_datasets.py` | ICC sweeps across datasets. |
| `eval_dl_comparison_table.py` | Daily-living comparison table assembly. |
| `eval_fogathome_finetune_curve.py`, `eval_fogathome_lopo_curve.py`, `eval_fogathome_segmentation.py`, `eval_fogathome_dailyliving.py` | FogAtHome fine-tuning / LOPO / segmentation curves (Fig 12). |
| `eval_embedding_probe.py`, `eval_knn.py` | Representation probes (Fig S1). |

> Scripts not listed as **Canonical** are one-off analyses or earlier iterations retained for
> reproducibility/provenance. When a result appears in the paper, trace it through
> `release/manifest.yaml > results` to the exact script + command.

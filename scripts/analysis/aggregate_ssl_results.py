"""
Aggregate SSL experiment results from kfold training log files.

Parses METRICS_JSON lines with stage="test", groups results per fold (in order
of appearance), computes mean ± std across folds, and prints a comparison table.

Usage:
    uv run python scripts/aggregate_ssl_results.py [--logs-dir logs/]
"""

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

@dataclass
class ExperimentDef:
    """Definition of one experiment row in the comparison table."""
    name: str                   # Display name
    patterns: list[str]         # Glob patterns to try (first match wins)
    labels_used: str            # e.g. "none", "defog", "defog + tdcsfog"
    exp_id: str                 # e.g. "016"
    description: str = ""       # Optional notes


EXPERIMENTS: list[ExperimentDef] = [
    ExperimentDef(
        name="Random init (linear probe)",
        patterns=["ablation_random_init*.log"],
        labels_used="none",
        exp_id="017",
        description="SpectralPatch backbone, random weights, frozen, GRU head",
    ),
    ExperimentDef(
        name="MAE linear probe",
        patterns=[
            "kfold_fogstrat_2026*.log",            # exp016 canonical run (fogstrat splits)
            "kfold_spectral_mae_defog*.log",
            "mae_finetune_defog_kfold*.log",
            "spectral_mae_linear_probe*.log",
        ],
        labels_used="none",
        exp_id="016",
        description="SpectralPatch pretrained on daily-living data, frozen backbone",
    ),
    ExperimentDef(
        name="MAE full finetune",
        patterns=[
            "full_finetune_20260404_180216.log",   # exp018 canonical run
            "full_finetune_2026*.log",
            "mae_full_finetune_defog_kfold*.log",
        ],
        labels_used="none",
        exp_id="018",
        description="SpectralPatch pretrained on daily-living data, differential LR",
    ),
    ExperimentDef(
        name="JEPA linear probe",
        patterns=[
            "jepa_finetune_defog_kfold_v2.log",
            "jepa_finetune*kfold*.log",
        ],
        labels_used="none",
        exp_id="019",
        description="JEPA pretrained on daily-living data, frozen backbone",
    ),
    ExperimentDef(
        name="Mixed supervised",
        patterns=["mixed_supervised_defog_kfold*.log"],
        labels_used="defog + tdcsfog",
        exp_id="020",
        description="SpectralPatch trained on defog + tdcsfog labeled data",
    ),
    ExperimentDef(
        name="SimCLR linear probe",
        patterns=["simclr_finetune*kfold*.log", "simclr_finetune_defog_kfold*.log"],
        labels_used="none",
        exp_id="021",
        description="SimCLR (SpectralPatchEncoder) pretrained on daily-living data",
    ),
    ExperimentDef(
        name="Causal MAE linear probe",
        patterns=["causal_mae_finetune*kfold*.log"],
        labels_used="none",
        exp_id="022",
        description="Causal next-patch MAE pretrained on daily-living data",
    ),
]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_METRICS_RE = re.compile(r"METRICS_JSON:\s*(\{.*\})")


def parse_test_ap_from_log(log_path: Path) -> list[float]:
    """
    Extract per-fold test AP values from a log file.

    Each fold produces exactly one METRICS_JSON entry with stage="test".
    They appear in order of fold execution, so position == fold index.

    Returns a list of AP values, one per fold.  An empty list means no test
    metrics were found (log may be incomplete / still running).
    """
    ap_values: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _METRICS_RE.search(line)
            if m is None:
                continue
            try:
                metrics = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if metrics.get("stage") != "test":
                continue
            if "ap" not in metrics:
                continue
            ap_values.append(float(metrics["ap"]))
    return ap_values


def find_log_file(logs_dir: Path, patterns: list[str]) -> Optional[Path]:
    """
    Return the first existing log file matched by any of the glob patterns.

    When multiple files match the same pattern (e.g. timestamped runs),
    the most-recently-modified file is returned.
    """
    for pattern in patterns:
        candidates = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    exp: ExperimentDef
    log_path: Optional[Path]
    fold_aps: list[float] = field(default_factory=list)

    @property
    def num_folds(self) -> int:
        return len(self.fold_aps)

    @property
    def mean(self) -> Optional[float]:
        if not self.fold_aps:
            return None
        return statistics.mean(self.fold_aps)

    @property
    def std(self) -> Optional[float]:
        if len(self.fold_aps) < 2:
            return None
        return statistics.stdev(self.fold_aps)

    @property
    def status(self) -> str:
        if self.log_path is None:
            return "pending"
        if not self.fold_aps:
            return "no_test_metrics"
        if self.num_folds < 5:
            return f"partial ({self.num_folds}/5)"
        return "complete"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_ap(ap: Optional[float]) -> str:
    if ap is None:
        return "  —  "
    return f"{ap:.3f}"


def _mean_std_str(result: ExperimentResult) -> str:
    if result.mean is None:
        return "pending" if result.log_path is None else "—"
    mean_str = f"{result.mean:.3f}"
    if result.std is not None:
        return f"{mean_str} ± {result.std:.3f}"
    return f"{mean_str}"


def build_plain_table(results: list[ExperimentResult]) -> str:
    """Build a plain-text table for terminal output."""
    header_name = "Experiment"
    header_folds = " F0    F1    F2    F3    F4 "
    header_mean = " Mean ± Std  "
    header_labels = "Labels"
    header_id = "ID "

    # Column widths
    name_w = max(len(header_name), max(len(r.exp.name) for r in results)) + 1

    lines: list[str] = []
    title = "SSL Experiment Comparison — Defog Domain (kfold_defog_fogstrat)"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(
        f"{'Experiment':<{name_w}}| {header_id} |{header_folds}| {header_mean}| {header_labels}"
    )
    sep_folds = "-" * len(header_folds)
    lines.append(
        f"{'-' * name_w}|-----|{sep_folds}|{'-' * len(header_mean)}|{'-' * (len(header_labels) + 2)}"
    )

    for r in results:
        # Build per-fold cell (always 5 slots)
        fold_cells = []
        for i in range(5):
            ap = r.fold_aps[i] if i < len(r.fold_aps) else None
            fold_cells.append(_fmt_ap(ap))
        folds_str = " ".join(fold_cells)

        mean_std = _mean_std_str(r)
        name_cell = f"{r.exp.name:<{name_w}}"
        id_cell = f"{r.exp.exp_id:>3}"
        lines.append(
            f"{name_cell}| {id_cell} | {folds_str} | {mean_std:<{len(header_mean) - 2}} | {r.exp.labels_used}"
        )

    return "\n".join(lines)


def build_markdown_table(results: list[ExperimentResult]) -> str:
    """Build a markdown table for saving to research/ssl_results_table.md."""
    lines: list[str] = []
    lines.append("# SSL Experiment Comparison — Defog Domain")
    lines.append("")
    lines.append("Test set: 100% defog patients. Splits: `kfold_defog_fogstrat`.")
    lines.append("Primary metric: Average Precision (AP), higher is better.")
    lines.append("")

    # Header
    lines.append("| ID  | Experiment | F0 | F1 | F2 | F3 | F4 | Mean ± Std | Labels used |")
    lines.append("|-----|------------|----|----|----|----|----|-----------:|-------------|")

    for r in results:
        fold_cells = []
        for i in range(5):
            ap = r.fold_aps[i] if i < len(r.fold_aps) else None
            fold_cells.append(_fmt_ap(ap))

        mean_std = _mean_std_str(r)
        cells = " | ".join(fold_cells)
        lines.append(
            f"| {r.exp.exp_id} | {r.exp.name} | {cells} | {mean_std} | {r.exp.labels_used} |"
        )

    lines.append("")
    lines.append(f"*Generated by `scripts/aggregate_ssl_results.py`.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-fold test AP from SSL experiment log files."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/"),
        help="Directory containing .log files (default: logs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/ssl_results_table.md"),
        help="Path to save the markdown table (default: research/ssl_results_table.md)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print table only; do not write markdown file",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    if not logs_dir.exists():
        print(f"ERROR: logs directory not found: {logs_dir}")
        raise SystemExit(1)

    # Collect results
    results: list[ExperimentResult] = []
    for exp in EXPERIMENTS:
        log_path = find_log_file(logs_dir, exp.patterns)
        result = ExperimentResult(exp=exp, log_path=log_path)
        if log_path is not None:
            result.fold_aps = parse_test_ap_from_log(log_path)
        results.append(result)

    # Print plain table
    print()
    print(build_plain_table(results))
    print()

    # Print per-experiment status summary
    print("Log file resolution:")
    for r in results:
        if r.log_path is None:
            resolved = "(not found)"
        else:
            resolved = str(r.log_path)
        print(f"  [{r.status:>16}]  {r.exp.name:<35}  {resolved}")
    print()

    # Save markdown table
    if not args.no_save:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(build_markdown_table(results), encoding="utf-8")
        print(f"Saved markdown table to: {output_path}")


if __name__ == "__main__":
    main()

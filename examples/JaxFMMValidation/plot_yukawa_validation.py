"""Build summary plots for Yukawa validation results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_YUKAWA_CONFIG
from opendust_jax.plotting import plot_force_scatter


def _load_case_metrics(output_dir: Path) -> list[dict]:
    metrics = []
    if (output_dir / "summary.json").exists():
        with (output_dir / "summary.json").open("r", encoding="utf-8") as summary_file:
            return json.load(summary_file)

    for metrics_file in sorted(output_dir.glob("rD_*_m/metrics.json")):
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics.append(json.load(handle))
    return metrics


def _flatten_attempts(cases: list[dict]) -> list[dict]:
    attempts = []
    for case in cases:
        for attempt_index, attempt in enumerate(case.get("attempts", [case])):
            row = dict(attempt)
            row["attempt_index"] = attempt_index
            row["debye_radius_m"] = case["debye_radius_m"]
            row["kappa_1_per_m"] = case["kappa_1_per_m"]
            row["case_accepted"] = case["accepted"]
            attempts.append(row)
    return attempts


def _plot_attempt_errors(attempts: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    backends = sorted({row["backend"] for row in attempts})
    markers = {"spherical": "o", "taylor": "s", "chebyshev": "^"}
    for backend in backends:
        rows = [row for row in attempts if row["backend"] == backend]
        x = [row["debye_radius_m"] for row in rows]
        y = [max(row["relative_l2"], 1e-16) for row in rows]
        labels = [f"p={row['p']}, theta={row['theta']}" for row in rows]
        ax.scatter(x, y, s=70, marker=markers.get(backend, "o"), label=backend)
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(DEFAULT_YUKAWA_CONFIG.acceptance_relative_l2, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Debye radius r_D, m")
    ax.set_ylabel("Relative L2 force error")
    ax.set_title("Yukawa validation accuracy by backend")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_timing(cases: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{case['backend']}\nr_D={case['debye_radius_m']:.1e}" for case in cases]
    tree_times = [case["tree_build_s"] for case in cases]
    eval_times = [case["yukawa_fmm_force_s"] for case in cases]
    direct_times = [case["direct_reference_s"] for case in cases]
    x = np.arange(len(cases))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    width = 0.25
    ax.bar(x - width, tree_times, width, label="tree build")
    ax.bar(x, eval_times, width, label="Yukawa FMM force")
    ax.bar(x + width, direct_times, width, label="direct reference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time, s")
    ax.set_title("Yukawa validation timing for selected candidates")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_summary_table(cases: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for case in cases:
        rows.append(
            [
                f"{case['debye_radius_m']:.2e}",
                case["backend"],
                str(case["p"]),
                f"{case['theta']:.2f}",
                f"{case['relative_l2']:.3e}",
                f"{case['yukawa_fmm_force_s']:.3f}",
                "PASS" if case["accepted"] else "FAIL",
            ]
        )

    fig, ax = plt.subplots(figsize=(10.0, 1.2 + 0.45 * len(rows)), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=("r_D, m", "backend", "p", "theta", "rel L2", "FMM s", "status"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax.set_title("Yukawa validation summary")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _regenerate_force_scatters(cases: list[dict], output_dir: Path) -> None:
    for case in cases:
        case_name = f"rD_{case['debye_radius_m']:.6e}_m".replace("+", "")
        case_dir = output_dir / case_name
        direct_file = case_dir / "forces_direct.npy"
        fmm_file = case_dir / "forces_yukawa_fmm.npy"
        if not direct_file.exists() or not fmm_file.exists():
            continue
        direct = np.load(direct_file)
        fmm = np.load(fmm_file)
        title = (
            f"Yukawa validation: backend={case['backend']}, p={case['p']}, "
            f"theta={case['theta']}, r_D={case['debye_radius_m']:.2e} m, "
            f"rel L2={case['relative_l2']:.3e}"
        )
        plot_force_scatter(direct, fmm, case_dir / "force_scatter_regenerated.png", title=title)


def main() -> None:
    output_dir = Path(DEFAULT_YUKAWA_CONFIG.output_dir)
    plots_dir = output_dir / "plots"
    cases = _load_case_metrics(output_dir)
    if not cases:
        raise SystemExit(
            f"No Yukawa validation metrics found in {output_dir}. "
            "Run validate_yukawa.py first."
        )

    attempts = _flatten_attempts(cases)
    _plot_attempt_errors(attempts, plots_dir / "attempt_relative_l2.png")
    _plot_timing(cases, plots_dir / "selected_timing.png")
    _plot_summary_table(cases, plots_dir / "summary_table.png")
    _regenerate_force_scatters(cases, output_dir)

    print(f"Saved plots to {plots_dir}")
    for path in sorted(plots_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()

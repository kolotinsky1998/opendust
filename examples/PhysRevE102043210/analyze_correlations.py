#!/usr/bin/env python3
"""Analyze charge and force correlations in final article-reproduction runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.correlation_analysis import (  # noqa: E402
    build_observable_histories,
    estimate_correlated_mean,
    normalized_autocorrelation,
)
from opendust_jax.models import ELEMENTARY_CHARGE_C  # noqa: E402


CASE_ORDER = ("case_I", "case_II", "case_III")
STAGE_ORDER = (
    ("constant", 1),
    ("linear", 5),
    ("quadratic", 5),
)
OBSERVABLES = {
    "charge_C": {
        "title": "Dust charge",
        "label": "charge",
        "unit": "C",
        "display_factor": 1.0 / (1.0e4 * ELEMENTARY_CHARGE_C),
        "display_unit": r"$10^4 e$",
    },
    "total_z_N": {
        "title": "Total non-electric axial force",
        "label": "total force",
        "unit": "N",
        "display_factor": 1.0e12,
        "display_unit": "pN",
    },
    "orbit_z_N": {
        "title": "Orbit axial force",
        "label": "orbit force",
        "unit": "N",
        "display_factor": 1.0e12,
        "display_unit": "pN",
    },
    "collection_z_N": {
        "title": "Collection axial force",
        "label": "collection force",
        "unit": "N",
        "display_factor": 1.0e12,
        "display_unit": "pN",
    },
    "dust_dust_z_N": {
        "title": "Dust-dust axial force",
        "label": "dust-dust force",
        "unit": "N",
        "display_factor": 1.0e12,
        "display_unit": "pN",
    },
}
BRANCH_STYLE = {
    "constant": ("Constant", "o", "#1f77b4"),
    "linear": ("Linear, iteration 5", "s", "#ff7f0e"),
    "quadratic": ("Quadratic, iteration 5", "^", "#2ca02c"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/daniilkolotinsky/Downloads/results"),
        help="Root containing case_I, case_II, and case_III",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("correlation_analysis"),
        help="Directory for CSV, metadata, summary, and plots",
    )
    parser.add_argument(
        "--acf-max-plasma-periods",
        type=float,
        default=20.0,
        help="Maximum lag shown in ACF figures",
    )
    return parser.parse_args()


def selected_artifacts(results_root: Path) -> list[tuple[str, str, int, Path]]:
    artifacts = []
    for case_name in CASE_ORDER:
        for branch, iteration in STAGE_ORDER:
            path = results_root / case_name / branch / f"iteration_{iteration:02d}.npz"
            metadata_path = path.with_suffix(".json")
            if not path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f"expected final artifact and metadata: {path}, {metadata_path}"
                )
            artifacts.append((case_name, branch, iteration, path))
    return artifacts


def analyze_artifacts(
    artifacts: list[tuple[str, str, int, Path]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str, int], np.ndarray],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    autocorrelations: dict[tuple[str, str, str, int], np.ndarray] = {}
    inputs: list[dict[str, Any]] = []

    for case_name, branch, iteration, artifact_path in artifacts:
        metadata_path = artifact_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError(f"artifact is not complete: {metadata_path}")
        burn_in_step = int(metadata["burn_in_step"])
        dt_s = float(metadata["dt_s"])
        plasma_period_s = float(metadata["ion_plasma_period_s"])

        with np.load(artifact_path) as data:
            histories = build_observable_histories(data, burn_in_step)
            dust_z_sim_m = np.asarray(data["dust_z_sim_m"], dtype=float)
            dust_z_paper_mm = np.asarray(data["dust_z_paper_mm"], dtype=float)
            times_s = np.asarray(data["times_s"], dtype=float)

        if times_s.size < 2 or not np.allclose(
            np.diff(times_s), dt_s, rtol=1.0e-7, atol=1.0e-15
        ):
            raise ValueError(f"inconsistent time grid in {artifact_path}")
        if dust_z_sim_m.shape != (7,) or dust_z_paper_mm.shape != (7,):
            raise ValueError(f"expected seven dust particles in {artifact_path}")

        stage_name = f"{case_name}/{branch}/iteration_{iteration:02d}"
        inputs.append(
            {
                "stage": stage_name,
                "artifact": str(artifact_path.resolve()),
                "metadata": str(metadata_path.resolve()),
                "fingerprint": metadata.get("fingerprint"),
                "dt_s": dt_s,
                "ion_plasma_period_s": plasma_period_s,
                "burn_in_step": burn_in_step,
                "post_burn_in_samples": int(next(iter(histories.values())).shape[0]),
                "artifact_size_bytes": artifact_path.stat().st_size,
                "artifact_modified_at": datetime.fromtimestamp(
                    artifact_path.stat().st_mtime
                ).astimezone().isoformat(),
            }
        )

        for observable, history in histories.items():
            for dust_index in range(history.shape[1]):
                values = history[:, dust_index]
                estimate = estimate_correlated_mean(values, dt_s)
                rho = normalized_autocorrelation(values)
                autocorrelations[(case_name, branch, observable, dust_index)] = rho
                rows.append(
                    {
                        "stage": stage_name,
                        "case": case_name,
                        "branch": branch,
                        "iteration": iteration,
                        "dust_index": dust_index + 1,
                        "dust_z_sim_m": float(dust_z_sim_m[dust_index]),
                        "dust_z_paper_mm": float(dust_z_paper_mm[dust_index]),
                        "observable": observable,
                        "unit": OBSERVABLES[observable]["unit"],
                        "dt_s": dt_s,
                        "ion_plasma_period_s": plasma_period_s,
                        "burn_in_step": burn_in_step,
                        **estimate.to_dict(),
                    }
                )
    return rows, autocorrelations, inputs


def write_statistics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, plots_dir: Path, stem: str) -> None:
    figure.savefig(plots_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_means(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    for observable, specification in OBSERVABLES.items():
        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), sharey=True)
        for axis, case_name in zip(axes, CASE_ORDER, strict=True):
            for branch, _ in STAGE_ORDER:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["case"] == case_name
                        and row["branch"] == branch
                        and row["observable"] == observable
                    ),
                    key=lambda row: row["dust_index"],
                )
                factor = float(specification["display_factor"])
                means = np.asarray([row["mean"] for row in selected]) * factor
                lower = np.asarray([row["confidence_low"] for row in selected]) * factor
                upper = np.asarray([row["confidence_high"] for row in selected]) * factor
                label, marker, color = BRANCH_STYLE[branch]
                axis.errorbar(
                    [row["dust_index"] for row in selected],
                    means,
                    yerr=np.vstack((means - lower, upper - means)),
                    label=label,
                    marker=marker,
                    color=color,
                    linewidth=1.4,
                    capsize=2.5,
                )
            axis.axhline(0.0, color="0.65", linewidth=0.8)
            axis.grid(alpha=0.25)
            axis.set_title(case_name.replace("case_", "Case "))
            axis.set_xlabel("Dust particle index")
            axis.set_xticks(range(1, 8))
        axes[0].set_ylabel(
            f"{specification['title']}, {specification['display_unit']}"
        )
        axes[-1].legend(fontsize=8)
        figure.suptitle(
            "Mean with 95% CI corrected for serial correlation",
            fontsize=11,
        )
        save_figure(figure, plots_dir, f"mean_{observable}")


def plot_autocorrelations(
    rows: list[dict[str, Any]],
    autocorrelations: dict[tuple[str, str, str, int], np.ndarray],
    plots_dir: Path,
    max_plasma_periods: float,
) -> None:
    for observable, specification in OBSERVABLES.items():
        figure, axes = plt.subplots(3, 3, figsize=(13.5, 10.0), sharex=True, sharey=True)
        for case_index, case_name in enumerate(CASE_ORDER):
            for branch_index, (branch, _) in enumerate(STAGE_ORDER):
                axis = axes[case_index, branch_index]
                reference = next(
                    row
                    for row in rows
                    if row["case"] == case_name
                    and row["branch"] == branch
                    and row["observable"] == observable
                )
                lag_scale = reference["dt_s"] / reference["ion_plasma_period_s"]
                for dust_index in range(7):
                    rho = autocorrelations[(case_name, branch, observable, dust_index)]
                    maximum_lag = min(
                        rho.size - 1, int(math.ceil(max_plasma_periods / lag_scale))
                    )
                    lags = np.arange(maximum_lag + 1) * lag_scale
                    axis.plot(
                        lags,
                        rho[: maximum_lag + 1],
                        linewidth=0.8,
                        label=f"Dust {dust_index + 1}",
                    )
                axis.axhline(0.0, color="black", linewidth=0.7)
                axis.axhline(math.exp(-1.0), color="0.5", linewidth=0.7, linestyle="--")
                axis.grid(alpha=0.2)
                if case_index == 0:
                    axis.set_title(BRANCH_STYLE[branch][0])
                if branch_index == 0:
                    axis.set_ylabel(
                        f"{case_name.replace('case_', 'Case ')}\nACF"
                    )
                if case_index == 2:
                    axis.set_xlabel(r"Lag, $\tau_i$")
        axes[0, -1].legend(fontsize=6, ncol=2)
        figure.suptitle(f"{specification['title']}: normalized autocorrelation")
        save_figure(figure, plots_dir, f"acf_{observable}")


def plot_heatmaps(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    stage_order = [
        (case_name, branch)
        for case_name in CASE_ORDER
        for branch, _ in STAGE_ORDER
    ]
    stage_labels = [
        f"{case_name.removeprefix('case_')} {branch[:4]}"
        for case_name, branch in stage_order
    ]
    for field, title, transform, format_string in (
        (
            "integrated_autocorrelation_time_s",
            r"Integrated autocorrelation time, $\tau_{int}/\tau_i$",
            lambda row: row[field] / row["ion_plasma_period_s"],
            ".2f",
        ),
        (
            "effective_sample_size",
            r"Effective sample size, $N_{eff}$",
            lambda row: row[field],
            ".0f",
        ),
    ):
        figure, axes = plt.subplots(5, 1, figsize=(10.5, 14.0), constrained_layout=True)
        for axis, (observable, specification) in zip(
            axes, OBSERVABLES.items(), strict=True
        ):
            matrix = np.empty((len(stage_order), 7), dtype=float)
            for stage_index, (case_name, branch) in enumerate(stage_order):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["case"] == case_name
                        and row["branch"] == branch
                        and row["observable"] == observable
                    ),
                    key=lambda row: row["dust_index"],
                )
                matrix[stage_index] = [transform(row) for row in selected]
            image = axis.imshow(matrix, aspect="auto", cmap="viridis")
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    axis.text(
                        column_index,
                        row_index,
                        format(matrix[row_index, column_index], format_string),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=(
                            "white"
                            if matrix[row_index, column_index] < np.nanmedian(matrix)
                            else "black"
                        ),
                    )
            axis.set_title(specification["title"])
            axis.set_xticks(range(7), [str(index) for index in range(1, 8)])
            axis.set_yticks(range(9), stage_labels)
            axis.set_xlabel("Dust particle index")
            figure.colorbar(image, ax=axis, shrink=0.75)
        figure.suptitle(title, fontsize=13)
        save_figure(figure, plots_dir, f"heatmap_{field}")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    longest = sorted(
        rows,
        key=lambda row: row["integrated_autocorrelation_time_s"],
        reverse=True,
    )[:12]
    smallest_neff = sorted(rows, key=lambda row: row["effective_sample_size"])[:12]
    stationarity = [row for row in rows if row["stationarity_warning"]]
    low_neff = [row for row in rows if row["low_effective_sample_size_warning"]]

    lines = [
        "# Correlation-aware force and charge analysis",
        "",
        "The reported uncertainty is the uncertainty of the mean after "
        "serial-correlation correction, not the instantaneous standard deviation.",
        "",
        f"- Analyzed series: {len(rows)}",
        f"- Stationarity warnings: {len(stationarity)}",
        f"- Series with effective sample size below 20: {len(low_neff)}",
        "",
        "## Longest integrated autocorrelation times",
        "",
        "| Stage | Dust | Observable | tau_int (s) | tau_int / tau_i | N_eff |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in longest:
        lines.append(
            f"| {row['stage']} | {row['dust_index']} | {row['observable']} | "
            f"{row['integrated_autocorrelation_time_s']:.5e} | "
            f"{row['integrated_autocorrelation_time_s'] / row['ion_plasma_period_s']:.3f} | "
            f"{row['effective_sample_size']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Smallest effective sample sizes",
            "",
            "| Stage | Dust | Observable | N_eff | g | 95% CI half-width | Stationarity warning |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in smallest_neff:
        half_width = 0.5 * (row["confidence_high"] - row["confidence_low"])
        lines.append(
            f"| {row['stage']} | {row['dust_index']} | {row['observable']} | "
            f"{row['effective_sample_size']:.2f} | {row['statistical_inefficiency']:.2f} | "
            f"{half_width:.5e} | {row['stationarity_warning']} |"
        )
    if stationarity:
        lines.extend(
            [
                "",
                "## Stationarity warnings",
                "",
                "A warning means that the two post-burn-in half-means differ by "
                "more than 1.96 combined correlation-corrected standard errors. "
                "The data are retained, but their CI should be interpreted cautiously.",
                "",
            ]
        )
        for row in stationarity:
            lines.append(
                f"- `{row['stage']}`, dust {row['dust_index']}, "
                f"`{row['observable']}`: z={row['stationarity_z_score']:.2f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.acf_max_plasma_periods) or args.acf_max_plasma_periods <= 0:
        raise ValueError("--acf-max-plasma-periods must be finite and positive")
    artifacts = selected_artifacts(args.results_root.resolve())
    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows, autocorrelations, inputs = analyze_artifacts(artifacts)
    if len(rows) != 9 * 7 * len(OBSERVABLES):
        raise RuntimeError(f"expected 315 statistical rows, received {len(rows)}")

    write_statistics_csv(rows, output_dir / "correlation_statistics.csv")
    plot_means(rows, plots_dir)
    plot_autocorrelations(
        rows,
        autocorrelations,
        plots_dir,
        args.acf_max_plasma_periods,
    )
    plot_heatmaps(rows, plots_dir)
    write_summary(rows, output_dir / "summary.md")

    analysis_metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "results_root": str(args.results_root.resolve()),
        "output_dir": str(output_dir),
        "selection": {
            "cases": list(CASE_ORDER),
            "stages": [
                {"branch": branch, "iteration": iteration}
                for branch, iteration in STAGE_ORDER
            ],
            "dust_particles_per_stage": 7,
            "observables": list(OBSERVABLES),
        },
        "method": {
            "burn_in": "burn_in_step from each production metadata file",
            "autocorrelation": "unbiased FFT autocovariance normalized at lag zero",
            "integrated_time": "Geyer initial-positive sequence; tau_int = dt * (1/2 + sum rho_k)",
            "statistical_inefficiency": "g = max(1, 2 tau_int / dt)",
            "standard_error": "sample_std * sqrt(g / N)",
            "confidence_interval": "two-sided 95% Student-t interval with df=max(1, N_eff-1)",
            "stationarity": (
                "absolute split-half mean difference divided by combined "
                "correlation-corrected SEM; warning above 1.96"
            ),
            "low_effective_sample_size_threshold": 20.0,
            "acf_plot_maximum_plasma_periods": args.acf_max_plasma_periods,
        },
        "row_count": len(rows),
        "inputs": inputs,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Analyzed {len(inputs)} production artifacts and {len(rows)} series")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()

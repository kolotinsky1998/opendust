#!/usr/bin/env python3
"""Create publication-ready figures from OpenDustJAX plane diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import gaussian_filter


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("input", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--reference-density", type=float, default=1.0e14)
    result.add_argument("--dust-radius-m", type=float, default=58.8e-6)
    result.add_argument("--dust-z-debye", type=float, default=-1.0)
    result.add_argument("--dpi", type=int, default=200)
    return result


def dust_marker(axis, dust_z_mm: float, dust_radius_mm: float, color: str) -> None:
    axis.add_patch(
        Circle(
            (0.0, dust_z_mm),
            dust_radius_mm,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            zorder=10,
        )
    )


def decorate_map(axis, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("z (mm)")
    axis.set_aspect("equal")


def save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        x_m = np.asarray(data["x_m"], dtype=np.float64)
        z_m = np.asarray(data["z_m"], dtype=np.float64)
        potential = np.asarray(data["potential_V"], dtype=np.float64)
        density = np.asarray(data["ion_density_m3"], dtype=np.float64)
        sample_count = int(data["sample_count"])
        sample_steps = np.asarray(data["sample_steps"], dtype=np.int64)
        sample_times = np.asarray(data["sample_times_s"], dtype=np.float64)

    x_mm, z_mm = 1.0e3 * x_m, 1.0e3 * z_m
    debye_m = max(abs(x_m[0]), abs(x_m[-1])) / 3.0
    dust_z_mm = 1.0e3 * args.dust_z_debye * debye_m
    dust_radius_mm = 1.0e3 * args.dust_radius_m
    normalized_density = density / args.reference_density
    smoothed_density = gaussian_filter(normalized_density, sigma=1.25, mode="nearest")
    dphi_dz, dphi_dx = np.gradient(potential, z_m, x_m)
    electric_x, electric_z = -dphi_dx, -dphi_dz
    electric_magnitude = np.hypot(electric_x, electric_z)
    subtitle = (
        f"average of {sample_count} samples, steps "
        f"{sample_steps[0]}-{sample_steps[-1]}, "
        f"t={1e6 * sample_times[0]:.2f}-{1e6 * sample_times[-1]:.2f} us"
    )
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        }
    )

    limit = float(np.max(np.abs(potential)))
    fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    image = axis.pcolormesh(
        x_mm,
        z_mm,
        potential,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    contours = axis.contour(
        x_mm, z_mm, potential, levels=12, colors="black", linewidths=0.35, alpha=0.5
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    dust_marker(axis, dust_z_mm, dust_radius_mm, "black")
    decorate_map(axis, f"Time-averaged electric potential\n{subtitle}")
    fig.colorbar(image, ax=axis, label="Potential (V)")
    save(fig, args.output_dir / "potential-map.png", args.dpi)

    fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    density_cap = float(np.percentile(normalized_density, 99.0))
    image = axis.pcolormesh(
        x_mm,
        z_mm,
        normalized_density,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=density_cap,
    )
    dust_marker(axis, dust_z_mm, dust_radius_mm, "white")
    decorate_map(
        axis,
        f"Time-averaged ion density (linear, capped at p99)\n{subtitle}",
    )
    fig.colorbar(image, ax=axis, label=r"$n_i/n_0$")
    save(fig, args.output_dir / "ion-density-linear-map.png", args.dpi)

    fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    positive_density = density[density > 0.0]
    image = axis.pcolormesh(
        x_mm,
        z_mm,
        density,
        shading="auto",
        cmap="magma",
        norm=LogNorm(vmin=float(positive_density.min()), vmax=float(density.max())),
    )
    dust_marker(axis, dust_z_mm, dust_radius_mm, "white")
    decorate_map(axis, f"Time-averaged ion density (log scale)\n{subtitle}")
    fig.colorbar(image, ax=axis, label=r"Ion density $n_i$ (m$^{-3}$)")
    save(fig, args.output_dir / "ion-density-log-map.png", args.dpi)

    fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    smoothed_cap = float(np.percentile(smoothed_density, 99.0))
    image = axis.pcolormesh(
        x_mm,
        z_mm,
        smoothed_density,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=smoothed_cap,
    )
    density_contours = axis.contour(
        x_mm,
        z_mm,
        smoothed_density,
        levels=(1.0, 2.0, 3.0, 4.0, 5.0),
        colors="white",
        linewidths=0.55,
        alpha=0.75,
    )
    axis.clabel(density_contours, inline=True, fontsize=7, fmt="%.0f")
    dust_marker(axis, dust_z_mm, dust_radius_mm, "white")
    decorate_map(
        axis,
        f"Smoothed time-averaged ion density (Gaussian sigma=1.25 cells)\n{subtitle}",
    )
    fig.colorbar(image, ax=axis, label=r"$n_i/n_0$")
    save(fig, args.output_dir / "ion-density-smoothed-map.png", args.dpi)

    fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    positive_field = electric_magnitude[electric_magnitude > 0.0]
    image = axis.pcolormesh(
        x_mm,
        z_mm,
        electric_magnitude,
        shading="auto",
        cmap="plasma",
        norm=LogNorm(vmin=float(positive_field.min()), vmax=float(positive_field.max())),
    )
    axis.streamplot(
        x_mm,
        z_mm,
        electric_x,
        electric_z,
        density=1.0,
        color="white",
        linewidth=0.55,
        arrowsize=0.8,
    )
    dust_marker(axis, dust_z_mm, dust_radius_mm, "white")
    decorate_map(axis, r"Derived electric field $\mathbf{E}=-\nabla\Phi$")
    fig.colorbar(image, ax=axis, label=r"Electric-field magnitude (V m$^{-1}$)")
    save(fig, args.output_dir / "electric-field-map.png", args.dpi)

    axis_index = int(np.argmin(np.abs(x_m)))
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(z_mm, potential[:, axis_index], color="#2166ac", linewidth=2.0)
    axes[0].set_ylabel("Potential (V)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(z_mm, normalized_density[:, axis_index], color="#1a9850", linewidth=2.0)
    axes[1].set_ylabel(r"$n_i/n_0$")
    axes[1].set_xlabel("z (mm)")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.axvline(dust_z_mm, color="black", linestyle="--", linewidth=1.0, label="dust z")
    axes[0].legend(loc="best")
    fig.suptitle(f"Profiles along the cylinder axis (x=0)\n{subtitle}")
    save(fig, args.output_dir / "axis-profiles.png", args.dpi)

    selected_z = (dust_z_mm, 0.0, -dust_z_mm)
    selected_indices = [int(np.argmin(np.abs(z_mm - value))) for value in selected_z]
    labels = (r"$z=-r_D$ (dust plane)", r"$z=0$", r"$z=+r_D$")
    colors = ("#d73027", "#4575b4", "#1a9850")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    for index, label, color in zip(selected_indices, labels, colors):
        axes[0].plot(x_mm, potential[index], label=label, color=color, linewidth=2.0)
        axes[1].plot(
            x_mm, normalized_density[index], label=label, color=color, linewidth=2.0
        )
    axes[0].set_ylabel("Potential (V)")
    axes[1].set_ylabel(r"$n_i/n_0$")
    axes[1].set_xlabel("x (mm)")
    axes[0].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Horizontal profiles at selected z positions\n{subtitle}")
    save(fig, args.output_dir / "horizontal-profiles.png", args.dpi)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)
    image0 = axes[0, 0].pcolormesh(
        x_mm,
        z_mm,
        potential,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    decorate_map(axes[0, 0], "Potential (V)")
    dust_marker(axes[0, 0], dust_z_mm, dust_radius_mm, "black")
    fig.colorbar(image0, ax=axes[0, 0], label="V")
    image1 = axes[0, 1].pcolormesh(
        x_mm,
        z_mm,
        smoothed_density,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=smoothed_cap,
    )
    decorate_map(axes[0, 1], "Ion density")
    dust_marker(axes[0, 1], dust_z_mm, dust_radius_mm, "white")
    fig.colorbar(image1, ax=axes[0, 1], label=r"$n_i/n_0$")
    axes[1, 0].plot(z_mm, potential[:, axis_index], color="#2166ac", linewidth=2.0)
    axes[1, 0].axvline(dust_z_mm, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Axis potential (x=0)")
    axes[1, 0].set_xlabel("z (mm)")
    axes[1, 0].set_ylabel("Potential (V)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(
        z_mm, normalized_density[:, axis_index], color="#1a9850", linewidth=2.0
    )
    axes[1, 1].axvline(dust_z_mm, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Axis ion density (x=0)")
    axes[1, 1].set_xlabel("z (mm)")
    axes[1, 1].set_ylabel(r"$n_i/n_0$")
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(f"Piel2017 OpenDustJAX plane diagnostics\n{subtitle}")
    save(fig, args.output_dir / "calculation-summary.png", args.dpi)

    for name in (
        "potential-map.png",
        "ion-density-linear-map.png",
        "ion-density-log-map.png",
        "ion-density-smoothed-map.png",
        "electric-field-map.png",
        "axis-profiles.png",
        "horizontal-profiles.png",
        "calculation-summary.png",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()

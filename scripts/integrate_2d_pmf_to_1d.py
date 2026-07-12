#!/usr/bin/env python3
"""Integrate a gridded 2D PMF into marginal 1D PMFs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-integrate-2d-pmf-to-1d")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KB_KCAL = 0.0019872041


def read_2d_pmf_xvg(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue

            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number} has fewer than 3 columns")

            x_val = float(parts[0])
            y_val = float(parts[1])
            z_text = parts[2].lower()
            z_val = np.nan if z_text == "nan" else float(parts[2])
            rows.append((x_val, y_val, z_val))

    if not rows:
        raise ValueError(f"{path} does not contain numeric PMF rows")

    x_coords = np.asarray(sorted({row[0] for row in rows}), dtype=float)
    y_coords = np.asarray(sorted({row[1] for row in rows}), dtype=float)
    grid = np.full((x_coords.size, y_coords.size), np.nan, dtype=float)

    x_index = {value: idx for idx, value in enumerate(x_coords)}
    y_index = {value: idx for idx, value in enumerate(y_coords)}

    for x_val, y_val, z_val in rows:
        grid[x_index[x_val], y_index[y_val]] = z_val

    return x_coords, y_coords, grid


def infer_uniform_spacing(coords: np.ndarray, axis_name: str) -> float:
    if coords.size < 2:
        return 1.0

    deltas = np.diff(coords)
    reference = float(deltas[0])
    if not np.allclose(deltas, reference, atol=1e-8, rtol=0.0):
        raise ValueError(f"{axis_name} grid is not uniform; cannot use simple rectangular integration")
    return reference


def logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")

    vmax = float(np.max(values))
    if not np.isfinite(vmax):
        return float("-inf")
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def integrate_axis(
    pmf_grid: np.ndarray,
    temperature: float,
    axis_step: float,
    axis: int,
) -> np.ndarray:
    beta = 1.0 / (KB_KCAL * temperature)
    result = np.full(pmf_grid.shape[axis], np.nan, dtype=float)

    for idx in range(result.size):
        slice_values = pmf_grid[idx, :] if axis == 0 else pmf_grid[:, idx]
        finite = np.isfinite(slice_values)
        if not np.any(finite):
            continue

        log_terms = -beta * slice_values[finite]
        partition_log = logsumexp(log_terms) + np.log(axis_step)
        result[idx] = -KB_KCAL * temperature * partition_log

    finite_result = np.isfinite(result)
    if not np.any(finite_result):
        raise ValueError("Integrated 1D PMF is empty")

    result[finite_result] -= np.min(result[finite_result])
    return result


def write_1d_pmf(path: Path, axis_name: str, coords: np.ndarray, pmf: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(f"# 1D marginal PMF integrated from a 2D PMF along {axis_name}\n")
        handle.write(f"# Columns: {axis_name}_deg pmf_kcal_per_mol\n")
        handle.write(f'@    title "{axis_name} 1D marginal PMF from 2D PMF"\n')
        handle.write(f'@    xaxis  label "{axis_name} (deg)"\n')
        handle.write('@    yaxis  label "PMF (kcal/mol)"\n')
        handle.write("@TYPE xy\n")
        for coord, value in zip(coords, pmf):
            value_text = "nan" if np.isnan(value) else f"{value:.8f}"
            handle.write(f"{coord:10.4f} {value_text:>12}\n")


def plot_1d_pmfs(
    out_png: Path,
    out_pdf: Path,
    x_name: str,
    x_coords: np.ndarray,
    x_pmf: np.ndarray,
    y_name: str,
    y_coords: np.ndarray,
    y_pmf: np.ndarray,
    plot_vmax: float | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True, sharey=True)

    specs = (
        (axes[0], x_name, x_coords, x_pmf),
        (axes[1], y_name, y_coords, y_pmf),
    )

    ymax = 0.0
    for _, _, _, pmf in specs:
        finite = np.isfinite(pmf)
        if np.any(finite):
            ymax = max(ymax, float(np.max(pmf[finite])))
    if plot_vmax is None:
        plot_vmax = max(1.0, ymax * 1.05)

    for ax, axis_name, coords, pmf in specs:
        finite = np.isfinite(pmf)
        if np.any(finite):
            ax.plot(coords[finite], pmf[finite], color="#1f77b4", linewidth=2.0)
            ax.scatter(coords[finite], pmf[finite], color="#1f77b4", s=18, zorder=3)
        ax.set_xlabel(f"{axis_name} (deg)", fontsize=12)
        ax.set_title(f"{axis_name} 1D marginal PMF", fontsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(float(coords[0]), float(coords[-1]))

    axes[0].set_ylabel("PMF (kcal/mol)", fontsize=12)
    axes[0].set_ylim(0.0, float(plot_vmax))

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrate a gridded 2D PMF into x/y marginal 1D PMFs via Boltzmann summation."
    )
    parser.add_argument("pmf2d", help="Input 2D PMF XVG file with columns x y pmf")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    parser.add_argument("--x-name", default="phi", help="Label/name for the first PMF axis")
    parser.add_argument("--y-name", default="psi", help="Label/name for the second PMF axis")
    parser.add_argument("--outdir", help="Output directory; defaults to the input file directory")
    parser.add_argument("--out-prefix", help="Output prefix; defaults to the input file stem without .xvg")
    parser.add_argument("--plot-vmax", type=float, help="Y-axis maximum for the 1D PMF plot")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pmf2d_path = Path(args.pmf2d).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else pmf2d_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    out_prefix = args.out_prefix if args.out_prefix else pmf2d_path.name.removesuffix(".xvg")

    x_coords, y_coords, pmf_grid = read_2d_pmf_xvg(pmf2d_path)
    y_step = infer_uniform_spacing(y_coords, args.y_name)
    x_step = infer_uniform_spacing(x_coords, args.x_name)

    x_pmf = integrate_axis(pmf_grid=pmf_grid, temperature=args.temperature, axis_step=y_step, axis=0)
    y_pmf = integrate_axis(pmf_grid=pmf_grid, temperature=args.temperature, axis_step=x_step, axis=1)

    x_out = outdir / f"{out_prefix}_{args.x_name}_1d_integrated_pmf.xvg"
    y_out = outdir / f"{out_prefix}_{args.y_name}_1d_integrated_pmf.xvg"
    fig_png = outdir / f"{out_prefix}_1d_integrated_pmf.png"
    fig_pdf = outdir / f"{out_prefix}_1d_integrated_pmf.pdf"

    write_1d_pmf(x_out, args.x_name, x_coords, x_pmf)
    write_1d_pmf(y_out, args.y_name, y_coords, y_pmf)
    plot_1d_pmfs(
        fig_png,
        fig_pdf,
        args.x_name,
        x_coords,
        x_pmf,
        args.y_name,
        y_coords,
        y_pmf,
        args.plot_vmax,
    )

    print("[INFO] Integrated 2D PMF into 1D marginal PMFs")
    print(f"       input        : {pmf2d_path}")
    print(f"       temperature  : {args.temperature}")
    print(f"       {args.x_name} bins     : {x_coords.size} (step {x_step:g} deg)")
    print(f"       {args.y_name} bins     : {y_coords.size} (step {y_step:g} deg)")
    print(f"       {args.x_name} 1D PMF   : {x_out}")
    print(f"       {args.y_name} 1D PMF   : {y_out}")
    print(f"       figure       : {fig_png}")
    print(f"       figure       : {fig_pdf}")


if __name__ == "__main__":
    main()

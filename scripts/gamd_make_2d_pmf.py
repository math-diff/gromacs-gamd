#!/usr/bin/env python3
"""Build and plot a 2D GaMD PMF from GROMACS RMSD/Rg and gamd-reweight outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - optional runtime dependency
    gaussian_filter = None


KB_KCAL = 0.0019872041


def read_xvg_values(path: Path, value_col_1based: int) -> np.ndarray:
    values = []
    value_col = value_col_1based - 1

    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue

            parts = stripped.split()
            if len(parts) <= value_col:
                raise ValueError(f"{path}:{line_number} has fewer than {value_col_1based} columns")

            values.append(float(parts[value_col]))

    if not values:
        raise ValueError(f"{path} does not contain numeric XVG data")

    return np.asarray(values, dtype=float)


def read_reweight(path: Path, mode: str) -> tuple[int, np.ndarray, np.ndarray]:
    nst_values = []
    steps = []
    delta_v = []

    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 9:
                raise ValueError(f"{path}:{line_number} has fewer than 9 columns")

            nst = int(float(parts[0]))
            step = int(float(parts[1]))
            boost_p = float(parts[7])
            boost_d = float(parts[8])

            if mode == "dual":
                dv = boost_p + boost_d
            elif mode == "potential":
                dv = boost_p
            elif mode == "dihedral":
                dv = boost_d
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            nst_values.append(nst)
            steps.append(step)
            delta_v.append(dv)

    if not steps:
        raise ValueError(f"{path} does not contain numeric reweight rows")

    unique_nst = np.unique(np.asarray(nst_values, dtype=int))
    if unique_nst.size != 1:
        raise ValueError(f"{path} contains multiple reweight_nst values: {unique_nst}")

    return int(unique_nst[0]), np.asarray(steps, dtype=np.int64), np.asarray(delta_v, dtype=float)


def save_text_matrix(path: Path, array: np.ndarray, header: str, fmt: str) -> None:
    np.savetxt(path, array, fmt=fmt, header=header, comments="# ")


def ensure_compatible_pyrew(src: Path, dst: Path) -> None:
    text = src.read_text()
    text = text.replace("dtype=np.int)", "dtype=int)")
    text = text.replace("dtype=np.int #", "dtype=int #")
    text = text.replace("dtype=np.int)", "dtype=int)")
    text = text.replace("scipy.misc.factorial", "scipy.special.factorial")
    text = text.replace("normed=True", "density=True")
    dst.write_text(text)


def prepare_inputs(
    workdir: Path,
    outdir: Path,
    prefix: str,
    mode: str,
    temperature: float,
    rmsd_col: int,
    rg_col: int,
    cv_start_step: int | None,
    cv_stride_step: int | None,
    force: bool,
) -> tuple[Path, Path, Path, float, float]:
    input2d_path = outdir / f"{prefix}.input2d.dat"
    weight_path = outdir / f"{prefix}.weight.dat"
    check_path = outdir / f"{prefix}.check.dat"

    if input2d_path.exists() and weight_path.exists() and check_path.exists() and not force:
        print(f"[INFO] Reusing existing prepared files in {outdir}")
        rmsd_values = read_xvg_values(workdir / "rmsd.xvg", rmsd_col)
        rg_values = read_xvg_values(workdir / "rg.xvg", rg_col)
        return input2d_path, weight_path, check_path, float(np.min(rmsd_values)), float(np.max(rg_values))

    rmsd_values = read_xvg_values(workdir / "rmsd.xvg", rmsd_col)
    rg_values = read_xvg_values(workdir / "rg.xvg", rg_col)
    if rmsd_values.size != rg_values.size:
        raise ValueError(f"RMSD and Rg lengths differ: {rmsd_values.size} vs {rg_values.size}")

    reweight_nst, steps, delta_v = read_reweight(workdir / "gamd-reweight.dat", mode)
    stride = cv_stride_step if cv_stride_step is not None else reweight_nst
    if cv_start_step is not None:
        start_step = cv_start_step
        print(f"[INFO] Using user-specified CV absolute start step: {start_step}")
    else:
        start_step = int(steps[-1]) - stride * (rmsd_values.size - 1)
        print(
            "[INFO] Inferred CV absolute start step by tail alignment: "
            f"{start_step}"
        )

    delta_steps = steps - start_step
    divisible = (delta_steps % stride) == 0
    frame_index = np.full(steps.shape, -1, dtype=np.int64)
    frame_index[divisible] = delta_steps[divisible] // stride

    in_range = (frame_index >= 0) & (frame_index < rmsd_values.size)
    keep = divisible & in_range
    if not np.any(keep):
        raise ValueError(
            "No frames align between gamd-reweight.dat and the XVG files. "
            "Check cv_start_step and cv_stride_step."
        )

    kept_steps = steps[keep]
    kept_index = frame_index[keep]
    kept_rmsd = rmsd_values[kept_index]
    kept_rg = rg_values[kept_index]
    kept_delta_v = delta_v[keep]
    dropped_reweight_rows = int(steps.size - kept_steps.size)

    beta = 1.0 / (KB_KCAL * temperature)
    beta_delta_v = beta * kept_delta_v
    exp_beta_delta_v = np.exp(np.clip(beta_delta_v, -700, 700))

    save_text_matrix(
        input2d_path,
        np.column_stack((kept_rmsd, kept_rg)),
        "RMSD(nm) Rg(nm)",
        "%.8f",
    )
    save_text_matrix(
        weight_path,
        np.column_stack((beta_delta_v, exp_beta_delta_v, kept_delta_v)),
        "beta_dV exp(beta_dV) dV(kcal/mol)",
        "%.8f",
    )
    save_text_matrix(
        check_path,
        np.column_stack((kept_steps, kept_index, kept_rmsd, kept_rg, kept_delta_v)),
        "step frame_index RMSD_nm Rg_nm dV_kcal_per_mol",
        "%.8f",
    )

    print("[INFO] Prepared aligned inputs:")
    print(f"       matched rows: {kept_steps.size}")
    print(f"       dropped leading CV frames: {int(kept_index[0])}")
    print(f"       dropped trailing CV frames: {int(rmsd_values.size - kept_index[-1] - 1)}")
    print(f"       dropped reweight rows: {dropped_reweight_rows}")
    print(f"       first matched step: {int(kept_steps[0])}")
    print(f"       last matched step : {int(kept_steps[-1])}")
    print(f"       stride            : {stride}")
    print(f"       start step        : {start_step}")

    return input2d_path, weight_path, check_path, float(np.min(kept_rmsd)), float(np.max(kept_rg))


def run_pyrew(
    workdir: Path,
    outdir: Path,
    input2d_path: Path,
    weight_path: Path,
    disc_x: float,
    disc_y: float,
    cutoff: int,
    temperature: float,
    emax: float,
    pmf_variant: str,
    force: bool,
) -> Path:
    pmf_path = outdir / f"pmf-{pmf_variant}-{input2d_path.name}.xvg"
    if pmf_path.exists() and not force:
        print(f"[INFO] Reusing existing PMF file {pmf_path.name}")
        return pmf_path

    compat_pyrew = outdir / "PyReweighting-2D.compat.py"
    ensure_compatible_pyrew(workdir / "PyReweighting-2D.py", compat_pyrew)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    cmd = [
        sys.executable,
        compat_pyrew.name,
        "-input",
        input2d_path.name,
        "-weight",
        weight_path.name,
        "-job",
        "amdweight_CE",
        "-discX",
        str(disc_x),
        "-discY",
        str(disc_y),
        "-cutoff",
        str(cutoff),
        "-T",
        str(temperature),
        "-Emax",
        str(emax),
    ]
    print("[INFO] Running PyReweighting-2D.py")
    subprocess.run(cmd, cwd=outdir, env=env, check=True)
    if not pmf_path.exists():
        raise FileNotFoundError(f"Expected PMF output was not created: {pmf_path}")

    return pmf_path


def read_pmf_xvg(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue
            x, y, z = stripped.split()[:3]
            rows.append((float(x), float(y), float(z)))

    if not rows:
        raise ValueError(f"{path} does not contain PMF data")

    data = np.asarray(rows, dtype=float)
    x_vals = np.unique(data[:, 0])
    y_vals = np.unique(data[:, 1])
    grid = np.full((x_vals.size, y_vals.size), np.nan, dtype=float)

    x_index = {value: idx for idx, value in enumerate(x_vals)}
    y_index = {value: idx for idx, value in enumerate(y_vals)}
    for x, y, z in data:
        grid[x_index[x], y_index[y]] = z

    return x_vals, y_vals, grid


def make_edges(left_edges: np.ndarray, spacing: float) -> np.ndarray:
    if left_edges.size == 0:
        raise ValueError("Cannot build edges from an empty coordinate array")
    return np.concatenate([left_edges, [left_edges[-1] + spacing]])


def smooth_grid_for_display(grid: np.ndarray, sigma: float, fill_value: float) -> np.ndarray:
    if sigma <= 0.0 or gaussian_filter is None:
        return grid

    valid = np.isfinite(grid)
    if not np.any(valid):
        return grid

    values = np.where(valid, grid, fill_value)
    weights = valid.astype(float)

    smooth_values = gaussian_filter(values * weights, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")

    smoothed = values.copy()
    nonzero = smooth_weights > 1e-12
    smoothed[nonzero] = smooth_values[nonzero] / smooth_weights[nonzero]
    smoothed[~valid] = np.nan
    return smoothed


def plot_pmf(
    pmf_path: Path,
    outdir: Path,
    output_stem: str,
    disc_x: float,
    disc_y: float,
    plot_vmax: float,
    panel_label: str,
    smooth_sigma: float,
) -> tuple[Path, Path]:
    x_left, y_left, grid = read_pmf_xvg(pmf_path)
    scale = 10.0  # nm -> A
    x_left = x_left * scale
    y_left = y_left * scale
    disc_x = disc_x * scale
    disc_y = disc_y * scale

    x_edges = make_edges(x_left, disc_x)
    y_edges = make_edges(y_left, disc_y)
    x_centers = x_left + 0.5 * disc_x
    y_centers = y_left + 0.5 * disc_y

    display_grid = smooth_grid_for_display(grid, smooth_sigma, plot_vmax)
    masked = np.ma.masked_invalid(display_grid.T)
    masked = np.ma.masked_greater(masked, plot_vmax)

    levels_fill = np.linspace(0.0, plot_vmax, 17)
    levels_line = np.arange(0.0, plot_vmax + 0.5, 0.5)

    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    mesh = ax.contourf(
        x_centers,
        y_centers,
        masked,
        levels=levels_fill,
        cmap="jet",
        vmin=0.0,
        vmax=plot_vmax,
        extend="neither",
    )
    finite_grid = np.where(np.isfinite(display_grid), display_grid, np.nan)
    ax.contour(
        x_centers,
        y_centers,
        finite_grid.T,
        levels=levels_line,
        colors="black",
        linewidths=0.55,
        alpha=0.65,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, ticks=np.arange(0.0, plot_vmax + 1.0, 1.0), shrink=0.9)
    cbar.set_label("PMF\n(kcal/mol)", rotation=0, labelpad=24, fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=11, width=1.2, length=4)

    ax.set_xlabel("RMSD (Å)", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"$R_g$ (Å)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12, width=1.5, length=6)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    if panel_label:
        ax.text(-0.16, 1.03, panel_label, transform=ax.transAxes, fontsize=30, fontweight="bold", va="top")

    png_path = outdir / f"{output_stem}.png"
    pdf_path = outdir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 2D GaMD PMF figure from RMSD/Rg and gamd-reweight.dat")
    parser.add_argument("workdir", help="Directory containing PyReweighting-2D.py, gamd-reweight.dat, rmsd.xvg, and rg.xvg")
    parser.add_argument("--outdir", default="pmf2d", help="Output subdirectory, relative to workdir unless absolute")
    parser.add_argument("--prefix", default="rmsd_rg", help="Prefix for generated intermediate files")
    parser.add_argument("--mode", choices=["dual", "potential", "dihedral"], default="dual")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--rmsd-col", type=int, default=2, help="1-based XVG column for RMSD values")
    parser.add_argument("--rg-col", type=int, default=2, help="1-based XVG column for total Rg")
    parser.add_argument("--cv-start-step", type=int, default=None, help="Absolute MD step corresponding to the first XVG frame")
    parser.add_argument("--cv-stride-step", type=int, default=None, help="MD step stride between XVG frames")
    parser.add_argument("--disc-x", type=float, default=0.02, help="PyReweighting bin size for RMSD (nm)")
    parser.add_argument("--disc-y", type=float, default=0.02, help="PyReweighting bin size for Rg (nm)")
    parser.add_argument("--cutoff", type=int, default=10, help="PyReweighting histogram cutoff")
    parser.add_argument("--emax", type=float, default=20.0, help="Maximum PMF plotted and passed to PyReweighting")
    parser.add_argument("--pmf-variant", choices=["c1", "c2", "c3"], default="c2", help="Which PyReweighting cumulant PMF to plot")
    parser.add_argument("--plot-vmax", type=float, default=8.0, help="Maximum PMF value shown in the final figure (kcal/mol)")
    parser.add_argument("--panel-label", default="", help="Optional panel label, e.g. C")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma for display only")
    parser.add_argument("--force", action="store_true", help="Rebuild aligned inputs and rerun PyReweighting even if outputs exist")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = workdir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    input2d_path, weight_path, check_path, _, _ = prepare_inputs(
        workdir=workdir,
        outdir=outdir,
        prefix=args.prefix,
        mode=args.mode,
        temperature=args.temperature,
        rmsd_col=args.rmsd_col,
        rg_col=args.rg_col,
        cv_start_step=args.cv_start_step,
        cv_stride_step=args.cv_stride_step,
        force=args.force,
    )

    pmf_path = run_pyrew(
        workdir=workdir,
        outdir=outdir,
        input2d_path=input2d_path,
        weight_path=weight_path,
        disc_x=args.disc_x,
        disc_y=args.disc_y,
        cutoff=args.cutoff,
        temperature=args.temperature,
        emax=args.emax,
        pmf_variant=args.pmf_variant,
        force=args.force,
    )

    png_path, pdf_path = plot_pmf(
        pmf_path=pmf_path,
        outdir=outdir,
        output_stem=f"pmf2d_{args.pmf_variant}",
        disc_x=args.disc_x,
        disc_y=args.disc_y,
        plot_vmax=args.plot_vmax,
        panel_label=args.panel_label,
        smooth_sigma=args.smooth_sigma,
    )

    print("[INFO] Outputs:")
    print(f"       input2d : {input2d_path}")
    print(f"       weight  : {weight_path}")
    print(f"       check   : {check_path}")
    print(f"       pmf     : {pmf_path}")
    print(f"       figure  : {png_path}")
    print(f"       figure  : {pdf_path}")


if __name__ == "__main__":
    main()

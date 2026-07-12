#!/usr/bin/env python3
"""Combine multiple aligned GaMD trajectories into one 2D PMF."""

from __future__ import annotations

import argparse
from pathlib import Path

from gamd_make_2d_pmf import plot_pmf, prepare_inputs, run_pyrew


DEFAULT_WORKDIRS = [
    Path("/media/yyf/8TB/gromacs-gamd-jdj-05/1uao-unfold-ok/charmm-gui-7364924553/gromacs"),
    Path("/media/yyf/8TB/gromacs-gamd-jdj-05/1uao-unfold-ok-2/charmm-gui-7364924553/gromacs"),
    Path("/media/yyf/8TB/gromacs-gamd/1uao-unfold-ok/charmm-gui-7364924553/gromacs"),
    Path("/media/yyf/8TB/gromacs-gamd/1uao-unfold-ok-2/charmm-gui-7364924553/gromacs"),
]


def build_label(workdir: Path) -> str:
    parts = workdir.parts
    return f"{parts[-4]}__{parts[-3]}".replace("-", "_")


def count_numeric_rows(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def append_numeric_rows(src: Path, dst, prefix: str | None = None) -> int:
    written = 0
    with src.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if prefix is None:
                dst.write(stripped + "\n")
            else:
                dst.write(f"{prefix} {stripped}\n")
            written += 1
    return written


def find_pyrew_source(workdirs: list[Path]) -> Path:
    for workdir in workdirs:
        if (workdir / "PyReweighting-2D.py").exists():
            return workdir
    raise FileNotFoundError("Could not find PyReweighting-2D.py in any workdir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine multiple GaMD trajectories into one 2D PMF")
    parser.add_argument("workdirs", nargs="*", help="Trajectory directories; defaults to the four 1UAO directories under /media/yyf/8TB")
    parser.add_argument("--outdir", default="/media/yyf/8TB/2dpmf", help="Final combined output directory")
    parser.add_argument("--prefix", default="combined_rmsd_rg", help="Prefix for merged input/weight/check files")
    parser.add_argument("--mode", choices=["dual", "potential", "dihedral"], default="dual")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--rmsd-col", type=int, default=2)
    parser.add_argument("--rg-col", type=int, default=2)
    parser.add_argument("--cv-start-step", type=int, default=0, help="Absolute start step for every trajectory")
    parser.add_argument("--cv-stride-step", type=int, default=50, help="MD step stride between saved CV frames")
    parser.add_argument("--disc-x", type=float, default=0.02)
    parser.add_argument("--disc-y", type=float, default=0.02)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--emax", type=float, default=20.0)
    parser.add_argument("--pmf-variant", choices=["c1", "c2", "c3"], default="c2")
    parser.add_argument("--plot-vmax", type=float, default=8.0)
    parser.add_argument("--panel-label", default="C")
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workdirs = [Path(p).resolve() for p in args.workdirs] if args.workdirs else [p.resolve() for p in DEFAULT_WORKDIRS]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prepared_root = outdir / "prepared"
    prepared_root.mkdir(exist_ok=True)

    combined_input = outdir / f"{args.prefix}.input2d.dat"
    combined_weight = outdir / f"{args.prefix}.weight.dat"
    combined_check = outdir / f"{args.prefix}.check.dat"
    manifest = outdir / "manifest.txt"

    with combined_input.open("w") as input_handle, combined_weight.open("w") as weight_handle, combined_check.open("w") as check_handle, manifest.open("w") as manifest_handle:
        input_handle.write("# RMSD(nm) Rg(nm)\n")
        weight_handle.write("# beta_dV exp(beta_dV) dV(kcal/mol)\n")
        check_handle.write("# source step frame_index RMSD_nm Rg_nm dV_kcal_per_mol\n")

        total_rows = 0
        for workdir in workdirs:
            label = build_label(workdir)
            per_outdir = prepared_root / label
            per_outdir.mkdir(exist_ok=True)

            input2d_path, weight_path, check_path, _, _ = prepare_inputs(
                workdir=workdir,
                outdir=per_outdir,
                prefix=label,
                mode=args.mode,
                temperature=args.temperature,
                rmsd_col=args.rmsd_col,
                rg_col=args.rg_col,
                cv_start_step=args.cv_start_step,
                cv_stride_step=args.cv_stride_step,
                force=args.force,
            )

            input_rows = append_numeric_rows(input2d_path, input_handle)
            weight_rows = append_numeric_rows(weight_path, weight_handle)
            check_rows = append_numeric_rows(check_path, check_handle, prefix=label)
            if not (input_rows == weight_rows == check_rows):
                raise ValueError(
                    f"Row mismatch for {workdir}: input={input_rows}, weight={weight_rows}, check={check_rows}"
                )

            total_rows += input_rows
            manifest_handle.write(f"{label}\t{workdir}\trows={input_rows}\n")
            print(f"[INFO] {label}: appended {input_rows} aligned rows")

    print(f"[INFO] Total merged rows: {total_rows}")

    pyrew_source = find_pyrew_source(workdirs)
    pmf_path = run_pyrew(
        workdir=pyrew_source,
        outdir=outdir,
        input2d_path=combined_input,
        weight_path=combined_weight,
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

    print("[INFO] Combined outputs:")
    print(f"       input2d : {combined_input}")
    print(f"       weight  : {combined_weight}")
    print(f"       check   : {combined_check}")
    print(f"       manifest: {manifest}")
    print(f"       pmf     : {pmf_path}")
    print(f"       figure  : {png_path}")
    print(f"       figure  : {pdf_path}")


if __name__ == "__main__":
    main()

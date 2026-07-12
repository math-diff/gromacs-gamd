#!/usr/bin/env python3
"""Run per-trajectory PyReweighting diagnostics for GaMD 2D PMFs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gamd_make_2d_pmf import ensure_compatible_pyrew, read_pmf_xvg


DEFAULT_PREPARED_ROOT = Path("/media/yyf/8TB/2dpmf/prepared")
DEFAULT_OUTDIR = Path("/media/yyf/8TB/2dpmf/per_traj_diag")
DEFAULT_PYREW_SOURCE = Path(
    "/media/yyf/8TB/gromacs-gamd-jdj-05/1uao-unfold-ok/charmm-gui-7364924553/gromacs/PyReweighting-2D.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run amd_dV and CE PMF diagnostics for each prepared trajectory")
    parser.add_argument("--prepared-root", default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--pyrew-source", default=str(DEFAULT_PYREW_SOURCE))
    parser.add_argument("--disc-x", type=float, default=0.02)
    parser.add_argument("--disc-y", type=float, default=0.02)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--emax", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_minimum(path: Path) -> tuple[float, float, float]:
    x_vals, y_vals, grid = read_pmf_xvg(path)
    idx = grid.argmin()
    ix, iy = divmod(idx, grid.shape[1])
    return float(x_vals[ix]), float(y_vals[iy]), float(grid[ix, iy])


def run_pyrew_job(
    compat_pyrew: Path,
    workdir: Path,
    input_name: str,
    weight_name: str,
    job: str,
    disc_x: float,
    disc_y: float,
    cutoff: int,
    temperature: float,
    emax: float,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(compat_pyrew.resolve()),
        "-input",
        input_name,
        "-weight",
        weight_name,
        "-job",
        job,
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
    env = {"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp"}
    env.update(**{})
    return subprocess.run(cmd, cwd=workdir, env=env, check=True, capture_output=True, text=True)


def main() -> None:
    args = parse_args()
    prepared_root = Path(args.prepared_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    compat_pyrew = outdir / "PyReweighting-2D.compat.py"
    ensure_compatible_pyrew(Path(args.pyrew_source).resolve(), compat_pyrew)

    summary_path = outdir / "summary.tsv"
    with summary_path.open("w") as summary:
        summary.write(
            "label\trows\tanharm_all_dV\tc1_min_x_nm\tc1_min_y_nm\tc1_min_kcal\t"
            "c2_min_x_nm\tc2_min_y_nm\tc2_min_kcal\tstdout_log\n"
        )

        for prepared_dir in sorted(p for p in prepared_root.iterdir() if p.is_dir()):
            label = prepared_dir.name
            input_path = prepared_dir / f"{label}.input2d.dat"
            weight_path = prepared_dir / f"{label}.weight.dat"
            if not input_path.exists() or not weight_path.exists():
                print(f"[WARNING] Missing prepared inputs for {label}, skipping")
                continue

            traj_outdir = outdir / label
            traj_outdir.mkdir(exist_ok=True)

            traj_input = traj_outdir / input_path.name
            traj_weight = traj_outdir / weight_path.name
            if args.force or not traj_input.exists():
                traj_input.write_bytes(input_path.read_bytes())
            if args.force or not traj_weight.exists():
                traj_weight.write_bytes(weight_path.read_bytes())

            ce_log = traj_outdir / "amdweight_CE.log"
            dv_log = traj_outdir / "amd_dV.log"

            if args.force or not (traj_outdir / f"pmf-c2-{input_path.name}.xvg").exists():
                ce_result = run_pyrew_job(
                    compat_pyrew,
                    traj_outdir,
                    traj_input.name,
                    traj_weight.name,
                    "amdweight_CE",
                    args.disc_x,
                    args.disc_y,
                    args.cutoff,
                    args.temperature,
                    args.emax,
                )
                ce_log.write_text(ce_result.stdout + ("\nSTDERR\n" + ce_result.stderr if ce_result.stderr else ""))
            else:
                ce_result = None

            if args.force or not (traj_outdir / f"dV-anharm-2D-{input_path.name}.xvg").exists():
                dv_result = run_pyrew_job(
                    compat_pyrew,
                    traj_outdir,
                    traj_input.name,
                    traj_weight.name,
                    "amd_dV",
                    args.disc_x,
                    args.disc_y,
                    args.cutoff,
                    args.temperature,
                    args.emax,
                )
                dv_log.write_text(dv_result.stdout + ("\nSTDERR\n" + dv_result.stderr if dv_result.stderr else ""))
            else:
                dv_result = None

            if dv_result is None:
                dv_text = dv_log.read_text()
            else:
                dv_text = dv_result.stdout
            match = re.search(r"Anharmonicity of all dV = ([^\n]+)", dv_text)
            anharm = float(match.group(1).strip()) if match else float("nan")

            c1_min = find_minimum(traj_outdir / f"pmf-c1-{input_path.name}.xvg")
            c2_min = find_minimum(traj_outdir / f"pmf-c2-{input_path.name}.xvg")
            rows = sum(1 for line in input_path.open() if line.strip() and not line.startswith("#"))

            summary.write(
                f"{label}\t{rows}\t{anharm:.8f}\t"
                f"{c1_min[0]:.6f}\t{c1_min[1]:.6f}\t{c1_min[2]:.6f}\t"
                f"{c2_min[0]:.6f}\t{c2_min[1]:.6f}\t{c2_min[2]:.6f}\t"
                f"{dv_log.name}\n"
            )
            print(
                f"[INFO] {label}: rows={rows}, anharm={anharm:.6f}, "
                f"c1_min=({c1_min[0]:.2f},{c1_min[1]:.2f}), "
                f"c2_min=({c2_min[0]:.2f},{c2_min[1]:.2f})"
            )

    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()

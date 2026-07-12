#!/usr/bin/env python3
"""Rewrite XVG time columns using a fixed timestep."""

from __future__ import annotations

import argparse
from pathlib import Path


def rewrite_xvg_time(input_path: Path, output_path: Path, dt_ps: float, t0_ps: float) -> int:
    frame_index = 0

    with input_path.open() as src, output_path.open("w") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                dst.write(line)
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"{input_path} contains a numeric line with fewer than 2 columns: {line.rstrip()}")

            time_ps = t0_ps + frame_index * dt_ps
            new_line = [f"{time_ps:.7f}"] + parts[1:]
            dst.write(" ".join(new_line) + "\n")
            frame_index += 1

    return frame_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fix the time column in one or more GROMACS XVG files")
    parser.add_argument("inputs", nargs="+", help="Input XVG files")
    parser.add_argument("--dt-ps", type=float, default=0.1, help="Nominal frame spacing in ps")
    parser.add_argument("--t0-ps", type=float, default=0.0, help="Start time in ps")
    parser.add_argument(
        "--suffix",
        default=".fixed_time",
        help="Suffix inserted before the original extension (default: .fixed_time)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for outputs; defaults to each input file's own directory",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for input_name in args.inputs:
        input_path = Path(input_name).resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        if output_dir is None:
            output_path = input_path.with_name(f"{input_path.stem}{args.suffix}{input_path.suffix}")
        else:
            output_path = output_dir / f"{input_path.stem}{args.suffix}{input_path.suffix}"

        n_frames = rewrite_xvg_time(input_path, output_path, args.dt_ps, args.t0_ps)
        print(f"[INFO] {input_path} -> {output_path} ({n_frames} frames, dt={args.dt_ps} ps, t0={args.t0_ps} ps)")


if __name__ == "__main__":
    main()

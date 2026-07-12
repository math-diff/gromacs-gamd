#!/usr/bin/env python3
"""Assess whether per-bin dV samples are approximately Gaussian."""

from __future__ import annotations

import argparse
import math
import warnings
from array import array
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - optional runtime dependency
    scipy_stats = None


PASS = "PASS"
BORDERLINE = "BORDERLINE"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT"

CLASS_CODE = {PASS: 0, BORDERLINE: 1, FAIL: 2, INSUFFICIENT: 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess Gaussianity of dV within every 2D PMF bin"
    )
    parser.add_argument("-input", required=True, help="Two-column 2D CV file")
    parser.add_argument("-weight", required=True, help="Weight file with dV in column 3")
    parser.add_argument(
        "-discX",
        type=float,
        default=6.0,
        help="Bin width in X (PyReweighting-2D.py compatible, default: 6)",
    )
    parser.add_argument(
        "-discY",
        type=float,
        default=6.0,
        help="Bin width in Y (PyReweighting-2D.py compatible, default: 6)",
    )
    parser.add_argument("-Xdim", nargs=2, type=float, help="Explicit X min/max")
    parser.add_argument("-Ydim", nargs=2, type=float, help="Explicit Y min/max")
    parser.add_argument(
        "-cutoff",
        "--cutoff",
        type=int,
        default=10,
        help="Minimum occupancy to treat a bin as populated (default: 10)",
    )
    parser.add_argument(
        "-min-samples",
        "--min-samples",
        type=int,
        default=30,
        help="Minimum dV samples for a PASS/BORDERLINE/FAIL judgment (default: 30)",
    )
    parser.add_argument(
        "-alpha-threshold",
        "--alpha-threshold",
        type=float,
        default=0.05,
        help="Primary anharmonicity threshold for PASS (default: 0.05)",
    )
    parser.add_argument(
        "-alpha-fail-threshold",
        "--alpha-fail-threshold",
        type=float,
        default=0.10,
        help="Anharmonicity threshold for FAIL (default: 0.10)",
    )
    parser.add_argument(
        "-p-threshold",
        "--p-threshold",
        type=float,
        default=1e-3,
        help="P-value threshold for D'Agostino-Pearson normality test (default: 1e-3)",
    )
    parser.add_argument(
        "-hist-bins",
        "--hist-bins",
        type=int,
        default=50,
        help="Histogram bins used in the anharmonicity calculation (default: 50)",
    )
    parser.add_argument(
        "-dV-column",
        "--dV-column",
        type=int,
        default=3,
        help="1-based column index of dV in the weight file (default: 3)",
    )
    parser.add_argument(
        "-out-prefix",
        "--out-prefix",
        default="",
        help="Output prefix; defaults to ./<input_stem>",
    )
    return parser.parse_args()


def iter_numeric_rows(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue
            yield line_number, stripped.split()


def pyreweight_default_dim(
    input_path: Path,
    column_index: int,
    disc: float,
) -> tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")

    for _, parts in iter_numeric_rows(input_path):
        if len(parts) <= column_index:
            raise ValueError(
                f"{input_path} has fewer than {column_index + 1} columns on a numeric row"
            )
        value = float(parts[column_index])
        minimum = min(minimum, value)
        maximum = max(maximum, value)

    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError(f"No numeric rows found in {input_path}")

    # Match PyReweighting-2D.py exactly, including the use of int() for negatives.
    min_data = disc * (int(minimum / disc) - 1)
    max_data = disc * (int(maximum / disc) + 1)
    return float(min_data), float(max_data)


def resolve_dims(
    input_path: Path,
    args: argparse.Namespace,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if args.Xdim is not None:
        xdim = (float(args.Xdim[0]), float(args.Xdim[1]))
    else:
        xdim = pyreweight_default_dim(input_path, 0, args.discX)

    if args.Ydim is not None:
        ydim = (float(args.Ydim[0]), float(args.Ydim[1]))
    else:
        ydim = pyreweight_default_dim(input_path, 1, args.discY)

    return xdim, ydim


def assignbins(dim: tuple[float, float], disc: float) -> np.ndarray:
    minimum, maximum = dim
    return np.arange(minimum, maximum + disc, disc, dtype=float)


def trapezoid(y: np.ndarray, dx: float) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    return float(np.trapz(y, dx=dx))


def anharm(data: np.ndarray, hist_bins: int) -> float:
    if data.size < 2:
        return float("nan")

    var = float(np.var(data))
    if var <= 0.0:
        return 0.0

    hist, edges = np.histogram(data, bins=hist_bins, density=True)
    hist = hist + 1.0e-300
    dx = float(edges[1] - edges[0])
    s_observed = -1.0 * trapezoid(hist * np.log(hist), dx=dx)
    s_gaussian = 0.5 * math.log(2.0 * math.pi * math.e * var + 1.0e-300)
    alpha = s_gaussian - s_observed

    if math.isinf(alpha) or math.isnan(alpha):
        return float("nan")
    return float(alpha)


def compute_moments(data: np.ndarray) -> tuple[float, float, float, float, float]:
    n = data.size
    mean = float(np.mean(data))
    std = float(np.std(data))

    if n < 2 or std == 0.0:
        return mean, std, 0.0, 0.0, float("nan")

    if scipy_stats is not None:
        skew = float(scipy_stats.skew(data, bias=False))
        kurt = float(scipy_stats.kurtosis(data, fisher=True, bias=False))
        if n >= 20:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pvalue = float(scipy_stats.normaltest(data).pvalue)
        else:
            pvalue = float("nan")
    else:
        centered = (data - mean) / std
        skew = float(np.mean(centered**3))
        kurt = float(np.mean(centered**4) - 3.0)
        pvalue = float("nan")

    return mean, std, skew, kurt, pvalue


def classify_bin(
    n: int,
    std: float,
    alpha: float,
    skew: float,
    kurt: float,
    pvalue: float,
    args: argparse.Namespace,
) -> str:
    if n < args.cutoff or n < args.min_samples:
        return INSUFFICIENT
    if std == 0.0:
        return FAIL
    if not math.isnan(alpha) and (
        alpha >= args.alpha_fail_threshold or abs(skew) >= 1.0 or abs(kurt) >= 2.0
    ):
        return FAIL
    if (
        not math.isnan(pvalue)
        and pvalue < args.p_threshold
        and not math.isnan(alpha)
        and alpha >= args.alpha_threshold
    ):
        return FAIL
    if not math.isnan(alpha) and (
        alpha >= args.alpha_threshold or abs(skew) >= 0.5 or abs(kurt) >= 1.0
    ):
        return BORDERLINE
    if not math.isnan(pvalue) and pvalue < args.p_threshold:
        return BORDERLINE
    return PASS


def safe_prefix(input_path: Path, out_prefix: str) -> Path:
    if out_prefix:
        prefix = Path(out_prefix).expanduser()
        if not prefix.is_absolute():
            prefix = Path.cwd() / prefix
    else:
        prefix = Path.cwd() / input_path.stem
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return prefix


def format_value(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.8g}"


def write_grid_header(handle, title: str) -> None:
    handle.write(f"# x_left y_left {title}\n")
    handle.write("@TYPE xy\n")


def main() -> None:
    args = parse_args()

    if args.discX <= 0.0 or args.discY <= 0.0:
        raise ValueError("-discX and -discY must be positive")
    if args.cutoff < 1:
        raise ValueError("-cutoff must be >= 1")
    if args.min_samples < 1:
        raise ValueError("-min-samples must be >= 1")
    if args.hist_bins < 2:
        raise ValueError("-hist-bins must be >= 2")
    if args.dV_column < 1:
        raise ValueError("-dV-column must be >= 1")

    input_path = Path(args.input).resolve()
    weight_path = Path(args.weight).resolve()
    out_prefix = safe_prefix(input_path, args.out_prefix)

    xdim, ydim = resolve_dims(input_path, args)
    x_edges = assignbins(xdim, args.discX)
    y_edges = assignbins(ydim, args.discY)
    nbin_x = len(x_edges) - 1
    nbin_y = len(y_edges) - 1

    dv_column_index = args.dV_column - 1
    dv_bins = [array("d") for _ in range(nbin_x * nbin_y)]
    all_dv = array("d")

    input_iter = iter_numeric_rows(input_path)
    weight_iter = iter_numeric_rows(weight_path)

    processed_rows = 0
    assigned_rows = 0

    for (i_line, i_parts), (w_line, w_parts) in zip(input_iter, weight_iter):
        if len(i_parts) < 2:
            raise ValueError(f"{input_path}:{i_line} has fewer than 2 columns")
        if len(w_parts) <= dv_column_index:
            raise ValueError(
                f"{weight_path}:{w_line} has fewer than {args.dV_column} columns"
            )

        x = float(i_parts[0])
        y = float(i_parts[1])
        dv = float(w_parts[dv_column_index])
        all_dv.append(dv)

        ix = int((x - x_edges[0]) / args.discX)
        iy = int((y - y_edges[0]) / args.discY)
        if 0 <= ix < nbin_x and 0 <= iy < nbin_y:
            dv_bins[ix * nbin_y + iy].append(dv)
            assigned_rows += 1
        processed_rows += 1

    try:
        next(input_iter)
        raise ValueError("Input file has more numeric rows than weight file")
    except StopIteration:
        pass
    try:
        next(weight_iter)
        raise ValueError("Weight file has more numeric rows than input file")
    except StopIteration:
        pass

    table_path = out_prefix.with_name(out_prefix.name + ".gaussianity.tsv")
    summary_path = out_prefix.with_name(out_prefix.name + ".summary.txt")
    count_grid_path = out_prefix.with_name(out_prefix.name + ".count.xvg")
    alpha_grid_path = out_prefix.with_name(out_prefix.name + ".alpha.xvg")
    pvalue_grid_path = out_prefix.with_name(out_prefix.name + ".pvalue.xvg")
    class_grid_path = out_prefix.with_name(out_prefix.name + ".class.xvg")
    stat_grid_path = out_prefix.with_name(out_prefix.name + ".dV_stat.xvg")

    class_counts = {PASS: 0, BORDERLINE: 0, FAIL: 0, INSUFFICIENT: 0}
    populated_bins = 0
    judged_bins = 0
    result_rows = []

    with (
        table_path.open("w") as table,
        count_grid_path.open("w") as count_grid,
        alpha_grid_path.open("w") as alpha_grid,
        pvalue_grid_path.open("w") as pvalue_grid,
        class_grid_path.open("w") as class_grid,
        stat_grid_path.open("w") as stat_grid,
    ):
        table.write(
            "x_left\tx_right\ty_left\ty_right\tn\tmean_dV\tstd_dV\tskewness\t"
            "excess_kurtosis\talpha\tpvalue\tclass\n"
        )
        write_grid_header(count_grid, "count")
        write_grid_header(alpha_grid, "alpha")
        write_grid_header(pvalue_grid, "pvalue")
        write_grid_header(class_grid, "class_code")
        stat_grid.write(
            "# x_left y_left count mean_dV std_dV alpha pvalue class_code\n"
        )
        stat_grid.write("@TYPE xy\n")

        for ix in range(nbin_x):
            for iy in range(nbin_y):
                x_left = float(x_edges[ix])
                x_right = float(x_edges[ix + 1])
                y_left = float(y_edges[iy])
                y_right = float(y_edges[iy + 1])

                data = np.asarray(dv_bins[ix * nbin_y + iy], dtype=np.float64)
                n = int(data.size)

                if n >= args.cutoff:
                    populated_bins += 1

                if n == 0:
                    mean = std = skew = kurt = alpha = pvalue = float("nan")
                    classification = INSUFFICIENT
                else:
                    mean, std, skew, kurt, pvalue = compute_moments(data)
                    alpha = anharm(data, args.hist_bins)
                    classification = classify_bin(
                        n=n,
                        std=std,
                        alpha=alpha,
                        skew=skew,
                        kurt=kurt,
                        pvalue=pvalue,
                        args=args,
                    )

                if classification != INSUFFICIENT:
                    judged_bins += 1
                    result_rows.append(
                        {
                            "x_left": x_left,
                            "x_right": x_right,
                            "y_left": y_left,
                            "y_right": y_right,
                            "n": n,
                            "mean": mean,
                            "std": std,
                            "skew": skew,
                            "kurt": kurt,
                            "alpha": alpha,
                            "pvalue": pvalue,
                            "class": classification,
                        }
                    )

                class_counts[classification] += 1

                table.write(
                    "\t".join(
                        [
                            format_value(x_left),
                            format_value(x_right),
                            format_value(y_left),
                            format_value(y_right),
                            str(n),
                            format_value(mean),
                            format_value(std),
                            format_value(skew),
                            format_value(kurt),
                            format_value(alpha),
                            format_value(pvalue),
                            classification,
                        ]
                    )
                    + "\n"
                )
                count_grid.write(f"{x_left:.8f} {y_left:.8f} {n}\n")
                alpha_grid.write(f"{x_left:.8f} {y_left:.8f} {format_value(alpha)}\n")
                pvalue_grid.write(
                    f"{x_left:.8f} {y_left:.8f} {format_value(pvalue)}\n"
                )
                class_grid.write(
                    f"{x_left:.8f} {y_left:.8f} {CLASS_CODE[classification]}\n"
                )
                stat_grid.write(
                    f"{x_left:.8f} {y_left:.8f} {n} "
                    f"{format_value(mean)} {format_value(std)} {format_value(alpha)} "
                    f"{format_value(pvalue)} {CLASS_CODE[classification]}\n"
                )

    all_dv_array = np.asarray(all_dv, dtype=np.float64)
    overall_mean, overall_std, overall_skew, overall_kurt, overall_pvalue = compute_moments(
        all_dv_array
    )
    overall_alpha = anharm(all_dv_array, args.hist_bins)

    worst_alpha_rows = sorted(
        (row for row in result_rows if not math.isnan(row["alpha"])),
        key=lambda row: row["alpha"],
        reverse=True,
    )[:10]
    largest_pop_bins = sorted(result_rows, key=lambda row: row["n"], reverse=True)[:10]

    with summary_path.open("w") as summary:
        summary.write("PyReweighting-compatible 2D dV Gaussianity analysis\n")
        summary.write(f"input = {input_path}\n")
        summary.write(f"weight = {weight_path}\n")
        summary.write(f"discX = {args.discX}\n")
        summary.write(f"discY = {args.discY}\n")
        summary.write(f"Xdim = [{xdim[0]}, {xdim[1]}]\n")
        summary.write(f"Ydim = [{ydim[0]}, {ydim[1]}]\n")
        summary.write(f"nbin_x = {nbin_x}\n")
        summary.write(f"nbin_y = {nbin_y}\n")
        summary.write(f"total_bins = {nbin_x * nbin_y}\n")
        summary.write(f"processed_rows = {processed_rows}\n")
        summary.write(f"assigned_rows = {assigned_rows}\n")
        summary.write(f"populated_bins(n>={args.cutoff}) = {populated_bins}\n")
        summary.write(f"judged_bins(n>={max(args.cutoff, args.min_samples)}) = {judged_bins}\n")
        summary.write(f"PASS = {class_counts[PASS]}\n")
        summary.write(f"BORDERLINE = {class_counts[BORDERLINE]}\n")
        summary.write(f"FAIL = {class_counts[FAIL]}\n")
        summary.write(f"INSUFFICIENT = {class_counts[INSUFFICIENT]}\n")
        summary.write("\n")
        summary.write(f"overall_mean_dV = {format_value(overall_mean)}\n")
        summary.write(f"overall_std_dV = {format_value(overall_std)}\n")
        summary.write(f"overall_skewness = {format_value(overall_skew)}\n")
        summary.write(f"overall_excess_kurtosis = {format_value(overall_kurt)}\n")
        summary.write(f"overall_alpha = {format_value(overall_alpha)}\n")
        summary.write(f"overall_pvalue = {format_value(overall_pvalue)}\n")
        summary.write("\n")
        summary.write("Top 10 bins by anharmonicity\n")
        summary.write(
            "x_left\tx_right\ty_left\ty_right\tn\tmean_dV\tstd_dV\tskewness\t"
            "excess_kurtosis\talpha\tpvalue\tclass\n"
        )
        for row in worst_alpha_rows:
            summary.write(
                "\t".join(
                    [
                        format_value(row["x_left"]),
                        format_value(row["x_right"]),
                        format_value(row["y_left"]),
                        format_value(row["y_right"]),
                        str(row["n"]),
                        format_value(row["mean"]),
                        format_value(row["std"]),
                        format_value(row["skew"]),
                        format_value(row["kurt"]),
                        format_value(row["alpha"]),
                        format_value(row["pvalue"]),
                        row["class"],
                    ]
                )
                + "\n"
            )
        summary.write("\n")
        summary.write("Top 10 bins by occupancy\n")
        summary.write(
            "x_left\tx_right\ty_left\ty_right\tn\tmean_dV\tstd_dV\tskewness\t"
            "excess_kurtosis\talpha\tpvalue\tclass\n"
        )
        for row in largest_pop_bins:
            summary.write(
                "\t".join(
                    [
                        format_value(row["x_left"]),
                        format_value(row["x_right"]),
                        format_value(row["y_left"]),
                        format_value(row["y_right"]),
                        str(row["n"]),
                        format_value(row["mean"]),
                        format_value(row["std"]),
                        format_value(row["skew"]),
                        format_value(row["kurt"]),
                        format_value(row["alpha"]),
                        format_value(row["pvalue"]),
                        row["class"],
                    ]
                )
                + "\n"
            )

    print(f"[INFO] Processed rows: {processed_rows}")
    print(f"[INFO] Assigned rows: {assigned_rows}")
    print(
        "[INFO] Binning: "
        f"Xdim=[{xdim[0]}, {xdim[1]}] with {nbin_x} bins, "
        f"Ydim=[{ydim[0]}, {ydim[1]}] with {nbin_y} bins"
    )
    print(f"[INFO] Overall dV mean/std: {overall_mean:.6f} / {overall_std:.6f}")
    print(f"[INFO] Overall dV alpha: {format_value(overall_alpha)}")
    print(f"[INFO] Overall dV normaltest p-value: {format_value(overall_pvalue)}")
    print(f"[INFO] Populated bins (n >= {args.cutoff}): {populated_bins}")
    print(f"[INFO] Judged bins (n >= {max(args.cutoff, args.min_samples)}): {judged_bins}")
    for key in (PASS, BORDERLINE, FAIL, INSUFFICIENT):
        print(f"[INFO] {key}: {class_counts[key]}")
    print(f"[INFO] Output table: {table_path}")
    print(f"[INFO] Output summary: {summary_path}")
    print(f"[INFO] Output count grid: {count_grid_path}")
    print(f"[INFO] Output alpha grid: {alpha_grid_path}")
    print(f"[INFO] Output pvalue grid: {pvalue_grid_path}")
    print(f"[INFO] Output class grid: {class_grid_path}")
    print(f"[INFO] Output dV stat grid: {stat_grid_path}")


if __name__ == "__main__":
    main()

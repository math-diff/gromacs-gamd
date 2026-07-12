#!/usr/bin/env python3
"""Build phi/psi 2D PMF and support-aware 1D MFEP PMFs from CMD dihedral XVG files.

The 2D PMF is computed from paired phi/psi time-series samples. The two 1D PMFs
are then extracted from the 2D PMF by tracing lowest-free-energy paths with
continuity constraints:

- phi 1D PMF: use phi as the progression coordinate and trace psi(phi)
- psi 1D PMF: use psi as the progression coordinate and trace phi(psi)

Unlike a fully circular dihedral scan, phi is sometimes only meaningfully
sampled over part of the periodic range. To avoid forcing the path through an
unsampled gap, this script can detect the largest low-support gap on the chosen
progress coordinate, rotate that gap to the boundary, and trace an anchored
open MFEP across the sampled span instead of a closed loop.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cmd-dih-phi-psi-mfep-open")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - optional runtime dependency
    gaussian_filter = None


KB_KCAL = 0.0019872041
FULL_PERIOD_DEG = 360.0
LOWER_DEG = -180.0


@dataclass(frozen=True)
class TraceMode:
    mode: str
    support_threshold: float
    largest_gap_start: int
    largest_gap_length: int
    rotation_shift: int


def read_xvg_xy(path: Path, x_col_1based: int = 1, y_col_1based: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    x_col = x_col_1based - 1
    y_col = y_col_1based - 1

    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue

            parts = stripped.split()
            required_cols = max(x_col, y_col) + 1
            if len(parts) < required_cols:
                raise ValueError(f"{path}:{line_number} has fewer than {required_cols} columns")

            rows.append((float(parts[x_col]), float(parts[y_col])))

    if not rows:
        raise ValueError(f"{path} does not contain numeric XVG rows")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


def build_periodic_axis(bin_width: float) -> tuple[np.ndarray, np.ndarray, float]:
    n_bins = int(round(FULL_PERIOD_DEG / bin_width))
    if not np.isclose(n_bins * bin_width, FULL_PERIOD_DEG, atol=1e-8):
        raise ValueError(f"bin_width={bin_width} does not evenly divide 360 degrees")

    centers = LOWER_DEG + np.arange(n_bins, dtype=float) * bin_width
    origin = LOWER_DEG - 0.5 * bin_width
    edges = origin + np.arange(n_bins + 1, dtype=float) * bin_width
    return centers, edges, origin


def wrap_periodic(values: np.ndarray, origin: float) -> np.ndarray:
    return ((values - origin) % FULL_PERIOD_DEG) + origin


def ensure_matching_times(phi_time: np.ndarray, psi_time: np.ndarray, tol: float) -> None:
    if phi_time.size != psi_time.size:
        raise ValueError(f"phi/psi frame counts differ: {phi_time.size} vs {psi_time.size}")

    if not np.allclose(phi_time, psi_time, atol=tol, rtol=0.0):
        max_diff = float(np.max(np.abs(phi_time - psi_time)))
        raise ValueError(f"phi/psi time grids do not match (max abs diff = {max_diff:.6g})")


def compute_2d_pmf(
    phi_values: np.ndarray,
    psi_values: np.ndarray,
    bin_width: float,
    temperature: float,
    count_cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi_centers, phi_edges, phi_origin = build_periodic_axis(bin_width)
    psi_centers, psi_edges, psi_origin = build_periodic_axis(bin_width)

    phi_wrapped = wrap_periodic(phi_values, phi_origin)
    psi_wrapped = wrap_periodic(psi_values, psi_origin)

    counts, _, _ = np.histogram2d(phi_wrapped, psi_wrapped, bins=(phi_edges, psi_edges))
    counts = counts.astype(int)

    total_count = int(np.sum(counts))
    if total_count <= 0:
        raise ValueError("No samples were accumulated into the 2D histogram")

    probabilities = counts.astype(float) / float(total_count)
    pmf = np.full(probabilities.shape, np.nan, dtype=float)

    valid = counts >= count_cutoff
    if not np.any(valid):
        raise ValueError(
            f"No 2D bins survive count_cutoff={count_cutoff}. Try reducing --count-cutoff."
        )

    pmf[valid] = -KB_KCAL * temperature * np.log(probabilities[valid])
    pmf[valid] -= np.nanmin(pmf[valid])

    return phi_centers, psi_centers, counts, probabilities, pmf


def smooth_grid(
    grid: np.ndarray,
    sigma: float,
    wrap_rows: bool,
    wrap_cols: bool,
    preserve_mask: bool,
) -> np.ndarray:
    if sigma <= 0.0 or gaussian_filter is None:
        return grid

    valid = np.isfinite(grid)
    if not np.any(valid):
        return grid

    fill_value = float(np.nanmax(grid[valid]))
    values = np.where(valid, grid, fill_value)
    weights = valid.astype(float)
    mode = ("wrap" if wrap_rows else "nearest", "wrap" if wrap_cols else "nearest")

    smooth_values = gaussian_filter(values * weights, sigma=sigma, mode=mode)
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode=mode)

    smoothed = np.full_like(grid, fill_value if not preserve_mask else np.nan, dtype=float)
    nonzero = smooth_weights > 1e-12
    smoothed[nonzero] = smooth_values[nonzero] / smooth_weights[nonzero]
    if preserve_mask:
        smoothed[~valid] = np.nan
    else:
        smoothed[~nonzero] = fill_value
    return smoothed


def build_search_energy(
    pmf: np.ndarray,
    counts: np.ndarray,
    mfep_count_cutoff: int,
    mfep_smooth_sigma: float,
    wrap_rows: bool,
    wrap_cols: bool,
) -> np.ndarray:
    support = (counts >= mfep_count_cutoff) & np.isfinite(pmf)
    if not np.any(support):
        raise ValueError(
            f"No PMF bins satisfy mfep_count_cutoff={mfep_count_cutoff}. "
            "Try reducing --mfep-count-cutoff."
        )

    seed = np.full_like(pmf, np.nan, dtype=float)
    seed[support] = pmf[support]
    return smooth_grid(
        seed,
        sigma=mfep_smooth_sigma,
        wrap_rows=wrap_rows,
        wrap_cols=wrap_cols,
        preserve_mask=False,
    )


def find_largest_false_gap(mask: np.ndarray) -> tuple[int, int]:
    if mask.ndim != 1:
        raise ValueError("Gap detection expects a 1D boolean mask")

    n = mask.size
    unsupported = ~mask
    if not np.any(unsupported):
        return 0, 0
    if np.all(unsupported):
        return 0, n

    doubled = np.concatenate((unsupported, unsupported))
    best_start = 0
    best_len = 0
    current_start = None

    for idx, value in enumerate(doubled):
        if value:
            if current_start is None:
                current_start = idx
        elif current_start is not None:
            length = idx - current_start
            if length > best_len:
                best_start = current_start
                best_len = length
            current_start = None

    if current_start is not None:
        length = doubled.size - current_start
        if length > best_len:
            best_start = current_start
            best_len = length

    best_len = min(best_len, n)
    best_start %= n
    return best_start, best_len


def choose_trace_mode(
    marginal_counts: np.ndarray,
    support_fraction: float,
    open_gap_min_bins: int,
) -> TraceMode:
    if support_fraction < 0.0:
        raise ValueError("support_fraction must be >= 0")

    if marginal_counts.size == 0:
        raise ValueError("marginal_counts cannot be empty")

    max_count = float(np.max(marginal_counts))
    threshold = max(1.0, max_count * support_fraction)
    supported = marginal_counts >= threshold

    if not np.any(supported):
        threshold = 1.0
        supported = marginal_counts > 0

    gap_start, gap_length = find_largest_false_gap(supported)
    if gap_length >= open_gap_min_bins and gap_length < supported.size:
        rotation_shift = (gap_start + gap_length) % supported.size
        return TraceMode(
            mode="open",
            support_threshold=threshold,
            largest_gap_start=gap_start,
            largest_gap_length=gap_length,
            rotation_shift=rotation_shift,
        )

    return TraceMode(
        mode="circular",
        support_threshold=threshold,
        largest_gap_start=gap_start,
        largest_gap_length=gap_length,
        rotation_shift=0,
    )


def trace_circular_mfep(
    energy: np.ndarray,
    anchor: tuple[int, int],
    max_step: int,
    jump_penalty: float,
) -> tuple[np.ndarray, str]:
    n_rows, n_cols = energy.shape
    anchor_row, anchor_col = anchor
    inf = float("inf")
    best_cost = None
    best_direction = ""
    best_path = None

    directions = (
        ("inc", [(anchor_row + offset) % n_rows for offset in range(n_rows)]),
        ("dec", [(anchor_row - offset) % n_rows for offset in range(n_rows)]),
    )

    for direction_name, row_order in directions:
        dp = np.full((n_rows, n_cols), inf, dtype=float)
        prev = np.full((n_rows, n_cols), -1, dtype=int)
        dp[0, anchor_col] = 0.0

        for t in range(1, n_rows):
            row_index = row_order[t]
            for col_index in range(n_cols):
                best_local_cost = inf
                best_prev = -1
                for delta in range(-max_step, max_step + 1):
                    prev_col = (col_index - delta) % n_cols
                    prev_cost = dp[t - 1, prev_col]
                    if not np.isfinite(prev_cost):
                        continue

                    step_distance = min(abs(delta), n_cols - abs(delta))
                    candidate_cost = prev_cost + energy[row_index, col_index] + jump_penalty * (step_distance**2)
                    if candidate_cost < best_local_cost:
                        best_local_cost = candidate_cost
                        best_prev = prev_col

                dp[t, col_index] = best_local_cost
                prev[t, col_index] = best_prev

        total_cost = dp[-1, anchor_col]
        if not np.isfinite(total_cost):
            continue

        path_indices = np.full(n_rows, -1, dtype=int)
        col_index = anchor_col
        for t in range(n_rows - 1, -1, -1):
            row_index = row_order[t]
            path_indices[row_index] = col_index
            if t > 0:
                col_index = prev[t, col_index]
                if col_index < 0:
                    break

        if best_cost is None or total_cost < best_cost:
            best_cost = float(total_cost)
            best_direction = direction_name
            best_path = path_indices

    if best_path is None:
        raise ValueError("Failed to trace a circular MFEP on the search energy surface")

    return best_path, best_direction


def trace_ordered_segment(
    energy: np.ndarray,
    row_order: np.ndarray,
    start_col: int,
    max_step: int,
    jump_penalty: float,
) -> np.ndarray:
    n_cols = energy.shape[1]
    inf = float("inf")
    length = row_order.size

    if length == 0:
        return np.empty(0, dtype=int)
    if length == 1:
        return np.asarray([start_col], dtype=int)

    dp = np.full((length, n_cols), inf, dtype=float)
    prev = np.full((length, n_cols), -1, dtype=int)
    dp[0, start_col] = 0.0

    for t in range(1, length):
        row_index = int(row_order[t])
        for col_index in range(n_cols):
            best_local_cost = inf
            best_prev = -1
            for delta in range(-max_step, max_step + 1):
                prev_col = (col_index - delta) % n_cols
                prev_cost = dp[t - 1, prev_col]
                if not np.isfinite(prev_cost):
                    continue

                step_distance = min(abs(delta), n_cols - abs(delta))
                candidate_cost = prev_cost + energy[row_index, col_index] + jump_penalty * (step_distance**2)
                if candidate_cost < best_local_cost:
                    best_local_cost = candidate_cost
                    best_prev = prev_col

            dp[t, col_index] = best_local_cost
            prev[t, col_index] = best_prev

    end_col = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, end_col]):
        raise ValueError("Failed to trace an ordered MFEP segment")

    path = np.full(length, -1, dtype=int)
    col_index = end_col
    for t in range(length - 1, -1, -1):
        path[t] = col_index
        if t > 0:
            col_index = prev[t, col_index]
            if col_index < 0:
                raise ValueError("Broken predecessor chain while tracing an ordered MFEP segment")

    return path


def trace_anchored_open_mfep(
    energy: np.ndarray,
    anchor: tuple[int, int],
    row_start: int,
    row_end: int,
    max_step: int,
    jump_penalty: float,
) -> np.ndarray:
    n_rows, _ = energy.shape
    anchor_row, anchor_col = anchor
    path = np.full(n_rows, -1, dtype=int)
    path[anchor_row] = anchor_col

    if row_end > anchor_row:
        forward_rows = np.arange(anchor_row, row_end + 1, dtype=int)
        forward_path = trace_ordered_segment(
            energy=energy,
            row_order=forward_rows,
            start_col=anchor_col,
            max_step=max_step,
            jump_penalty=jump_penalty,
        )
        path[forward_rows] = forward_path

    if row_start < anchor_row:
        backward_rows = np.arange(anchor_row, row_start - 1, -1, dtype=int)
        backward_path = trace_ordered_segment(
            energy=energy,
            row_order=backward_rows,
            start_col=anchor_col,
            max_step=max_step,
            jump_penalty=jump_penalty,
        )
        path[backward_rows] = backward_path

    return path


def build_profile_from_path(
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    raw_pmf: np.ndarray,
    raw_counts: np.ndarray,
    raw_probabilities: np.ndarray,
    path_indices: np.ndarray,
) -> np.ndarray:
    profile = np.full((row_coords.size, 5), np.nan, dtype=float)
    profile[:, 0] = row_coords

    for row_index, col_index in enumerate(path_indices):
        if col_index < 0:
            continue

        profile[row_index, 1] = col_coords[col_index]
        if np.isfinite(raw_pmf[row_index, col_index]):
            profile[row_index, 2] = raw_pmf[row_index, col_index]
            profile[row_index, 3] = raw_counts[row_index, col_index]
            profile[row_index, 4] = raw_probabilities[row_index, col_index]
        else:
            profile[row_index, 3] = 0.0
            profile[row_index, 4] = 0.0

    return profile


def circular_index_distance(a: int, b: int, period: int) -> int:
    delta = abs(a - b)
    return min(delta, period - delta)


def project_path_segment_to_valid_bins(
    raw_pmf: np.ndarray,
    traced_path: np.ndarray,
    row_indices: np.ndarray,
    snap_deviation_penalty: float,
    snap_jump_penalty: float,
) -> np.ndarray:
    n_cols = raw_pmf.shape[1]
    inf = float("inf")

    candidate_cols_per_row: list[np.ndarray] = []
    for row_index in row_indices:
        candidate_cols = np.where(np.isfinite(raw_pmf[row_index]))[0]
        if candidate_cols.size == 0:
            raise ValueError("Cannot project MFEP segment because a row has no sampled bins")
        candidate_cols_per_row.append(candidate_cols.astype(int))

    dp: list[np.ndarray] = []
    prev: list[np.ndarray] = []

    for offset, row_index in enumerate(row_indices):
        target_col = int(traced_path[row_index])
        cols = candidate_cols_per_row[offset]
        costs = np.full(cols.size, inf, dtype=float)
        parents = np.full(cols.size, -1, dtype=int)

        if offset == 0:
            for i, col in enumerate(cols):
                deviation = circular_index_distance(int(col), target_col, n_cols)
                costs[i] = raw_pmf[row_index, col] + snap_deviation_penalty * (deviation**2)
        else:
            prev_cols = candidate_cols_per_row[offset - 1]
            prev_costs = dp[offset - 1]
            for i, col in enumerate(cols):
                deviation = circular_index_distance(int(col), target_col, n_cols)
                best_cost = inf
                best_parent = -1
                for j, prev_col in enumerate(prev_cols):
                    jump = circular_index_distance(int(col), int(prev_col), n_cols)
                    candidate = (
                        prev_costs[j]
                        + raw_pmf[row_index, col]
                        + snap_deviation_penalty * (deviation**2)
                        + snap_jump_penalty * (jump**2)
                    )
                    if candidate < best_cost:
                        best_cost = candidate
                        best_parent = j
                costs[i] = best_cost
                parents[i] = best_parent

        dp.append(costs)
        prev.append(parents)

    projected = np.full(row_indices.size, -1, dtype=int)
    choice = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1][choice]):
        raise ValueError("Failed to project MFEP segment onto sampled bins")

    for offset in range(row_indices.size - 1, -1, -1):
        projected[offset] = int(candidate_cols_per_row[offset][choice])
        if offset > 0:
            choice = int(prev[offset][choice])
            if choice < 0:
                raise ValueError("Broken predecessor chain while projecting MFEP segment")

    return projected


def project_path_to_sampled_bins(
    raw_pmf: np.ndarray,
    traced_path: np.ndarray,
    snap_deviation_penalty: float,
    snap_jump_penalty: float,
) -> np.ndarray:
    projected = traced_path.copy()
    row_has_support = np.any(np.isfinite(raw_pmf), axis=1)
    valid_rows = np.where((traced_path >= 0) & row_has_support)[0]
    if valid_rows.size == 0:
        projected[~row_has_support] = -1
        return projected

    starts = [valid_rows[0]]
    for i in range(1, valid_rows.size):
        if valid_rows[i] != valid_rows[i - 1] + 1:
            starts.append(valid_rows[i])

    ends = []
    for i in range(valid_rows.size - 1):
        if valid_rows[i + 1] != valid_rows[i] + 1:
            ends.append(valid_rows[i])
    ends.append(valid_rows[-1])

    for start, end in zip(starts, ends):
        row_indices = np.arange(start, end + 1, dtype=int)
        segment = project_path_segment_to_valid_bins(
            raw_pmf=raw_pmf,
            traced_path=traced_path,
            row_indices=row_indices,
            snap_deviation_penalty=snap_deviation_penalty,
            snap_jump_penalty=snap_jump_penalty,
        )
        projected[row_indices] = segment

    projected[~row_has_support] = -1
    return projected


def apply_marginal_support_mask(
    profile: np.ndarray,
    marginal_counts: np.ndarray,
    support_fraction: float,
) -> tuple[np.ndarray, float]:
    if support_fraction <= 0.0:
        return profile, 0.0

    threshold = float(np.max(marginal_counts)) * support_fraction
    masked = profile.copy()
    low_support = marginal_counts < threshold
    masked[low_support, 1:] = np.nan
    return masked, threshold


def extract_axis_mfep(
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    pmf: np.ndarray,
    counts: np.ndarray,
    probabilities: np.ndarray,
    anchor: tuple[int, int],
    gap_support_fraction: float,
    open_gap_min_bins: int,
    mfep_count_cutoff: int,
    mfep_smooth_sigma: float,
    mfep_max_step: int,
    mfep_jump_penalty: float,
    snap_deviation_penalty: float,
    snap_jump_penalty: float,
) -> tuple[np.ndarray, np.ndarray, TraceMode]:
    marginal_counts = np.sum(counts, axis=1)
    trace_mode = choose_trace_mode(
        marginal_counts=marginal_counts,
        support_fraction=gap_support_fraction,
        open_gap_min_bins=open_gap_min_bins,
    )

    if trace_mode.mode == "circular":
        search_energy = build_search_energy(
            pmf=pmf,
            counts=counts,
            mfep_count_cutoff=mfep_count_cutoff,
            mfep_smooth_sigma=mfep_smooth_sigma,
            wrap_rows=True,
            wrap_cols=True,
        )
        path_indices, _ = trace_circular_mfep(
            energy=search_energy,
            anchor=anchor,
            max_step=mfep_max_step,
            jump_penalty=mfep_jump_penalty,
        )
    else:
        shift = trace_mode.rotation_shift
        pmf_rot = np.roll(pmf, -shift, axis=0)
        counts_rot = np.roll(counts, -shift, axis=0)
        anchor_rot = ((anchor[0] - shift) % pmf.shape[0], anchor[1])
        marginal_rot = np.roll(marginal_counts, -shift)
        support_rot = marginal_rot >= trace_mode.support_threshold
        support_rows = np.where(support_rot)[0]
        if support_rows.size == 0:
            raise ValueError("No rows remain after applying the open-path support threshold")

        row_start = int(support_rows[0])
        row_end = int(support_rows[-1])
        search_energy = build_search_energy(
            pmf=pmf_rot,
            counts=counts_rot,
            mfep_count_cutoff=mfep_count_cutoff,
            mfep_smooth_sigma=mfep_smooth_sigma,
            wrap_rows=False,
            wrap_cols=True,
        )
        path_rot = trace_anchored_open_mfep(
            energy=search_energy,
            anchor=anchor_rot,
            row_start=row_start,
            row_end=row_end,
            max_step=mfep_max_step,
            jump_penalty=mfep_jump_penalty,
        )
        path_indices = np.roll(path_rot, shift)

    path_indices = project_path_to_sampled_bins(
        raw_pmf=pmf,
        traced_path=path_indices,
        snap_deviation_penalty=snap_deviation_penalty,
        snap_jump_penalty=snap_jump_penalty,
    )

    profile = build_profile_from_path(
        row_coords=row_coords,
        col_coords=col_coords,
        raw_pmf=pmf,
        raw_counts=counts,
        raw_probabilities=probabilities,
        path_indices=path_indices,
    )
    return profile, path_indices, trace_mode


def write_paired_samples(path: Path, time: np.ndarray, phi: np.ndarray, psi: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("# Paired phi/psi samples used to build the PMF\n")
        handle.write("# Columns: time_ps phi_deg psi_deg\n")
        handle.write('@    title "Paired phi/psi dihedral samples"\n')
        handle.write('@    xaxis  label "Time (ps)"\n')
        handle.write('@    yaxis  label "Angle (deg)"\n')
        handle.write("@TYPE xy\n")
        for time_ps, phi_deg, psi_deg in zip(time, phi, psi):
            handle.write(f"{time_ps:12.5f} {phi_deg:12.6f} {psi_deg:12.6f}\n")


def write_2d_pmf(
    path: Path,
    phi_coords: np.ndarray,
    psi_coords: np.ndarray,
    pmf: np.ndarray,
    counts: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    with path.open("w") as handle:
        handle.write("# 2D phi/psi PMF derived from paired CMD dihedral samples\n")
        handle.write("# Columns: phi_deg psi_deg pmf_kcal_per_mol count probability\n")
        handle.write('@    title "phi/psi 2D PMF"\n')
        handle.write('@    xaxis  label "phi (deg)"\n')
        handle.write('@    yaxis  label "psi (deg)"\n')
        handle.write("@TYPE xyz\n")
        for i, phi_deg in enumerate(phi_coords):
            for j, psi_deg in enumerate(psi_coords):
                pmf_value = pmf[i, j]
                pmf_text = "nan" if np.isnan(pmf_value) else f"{pmf_value:.8f}"
                handle.write(
                    f"{phi_deg:10.4f} {psi_deg:10.4f} {pmf_text:>12} "
                    f"{counts[i, j]:8d} {probabilities[i, j]:.8e}\n"
                )
            handle.write("\n")


def write_1d_profile(path: Path, axis_name: str, partner_name: str, profile: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(
            f"# 1D {axis_name} PMF extracted along the support-aware 2D MFEP and reported on the raw 2D PMF\n"
        )
        handle.write(
            f"# Columns: {axis_name}_deg pmf_kcal_per_mol {partner_name}_deg_on_mfep count probability\n"
        )
        handle.write(f'@    title "{axis_name} 1D PMF from 2D MFEP"\n')
        handle.write(f'@    xaxis  label "{axis_name} (deg)"\n')
        handle.write('@    yaxis  label "PMF (kcal/mol)"\n')
        handle.write("@TYPE xy\n")

        for coord_deg, partner_deg, pmf_value, count, probability in profile:
            pmf_text = "nan" if np.isnan(pmf_value) else f"{pmf_value:.8f}"
            partner_text = "nan" if np.isnan(partner_deg) else f"{partner_deg:.4f}"
            count_text = "nan" if np.isnan(count) else f"{int(round(count))}"
            prob_text = "nan" if np.isnan(probability) else f"{probability:.8e}"
            handle.write(
                f"{coord_deg:10.4f} {pmf_text:>12} {partner_text:>10} "
                f"{count_text:>8} {prob_text:>14}\n"
            )


def profile_to_path_xy(
    x_coords: np.ndarray,
    partner_coords: np.ndarray,
    path_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = x_coords.astype(float).copy()
    y_values = np.full(x_coords.shape, np.nan, dtype=float)
    valid = path_indices >= 0
    y_values[valid] = partner_coords[path_indices[valid]]
    return x_values, y_values


def plot_path_segments(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid):
        return

    valid_index = np.where(valid)[0]
    starts = [0]
    starts.extend((np.where(np.diff(valid_index) > 1)[0] + 1).tolist())
    starts.append(valid_index.size)

    label_added = False
    for start, end in zip(starts[:-1], starts[1:]):
        segment_index = valid_index[start:end]
        if segment_index.size <= 1:
            continue

        local_starts = [0]
        jump_mask = (
            (np.abs(np.diff(x_values[segment_index])) > 180.0)
            | (np.abs(np.diff(y_values[segment_index])) > 180.0)
        )
        local_starts.extend((np.where(jump_mask)[0] + 1).tolist())
        local_starts.append(segment_index.size)

        for local_start, local_end in zip(local_starts[:-1], local_starts[1:]):
            final_index = segment_index[local_start:local_end]
            if final_index.size <= 1:
                continue
            ax.plot(
                x_values[final_index],
                y_values[final_index],
                color=color,
                linewidth=1.7,
                linestyle=linestyle,
                alpha=0.95,
                label=label if not label_added else "_nolegend_",
            )
            label_added = True


def plot_2d_pmf(
    out_png: Path,
    out_pdf: Path,
    phi_coords: np.ndarray,
    psi_coords: np.ndarray,
    pmf: np.ndarray,
    phi_path_indices: np.ndarray,
    psi_path_indices: np.ndarray,
    anchor_index: tuple[int, int],
    plot_vmax: float,
    smooth_sigma: float,
) -> None:
    display_grid = smooth_grid(pmf, sigma=smooth_sigma, wrap_rows=True, wrap_cols=True, preserve_mask=True)
    finite_mask = np.isfinite(display_grid)
    if not np.any(finite_mask):
        raise ValueError("Cannot plot 2D PMF because every bin is invalid")

    masked = np.ma.masked_invalid(display_grid.T)
    finite_display = np.where(np.isfinite(display_grid), display_grid, np.nan)

    fig, ax = plt.subplots(figsize=(6.6, 5.9), constrained_layout=True)
    levels_fill = np.linspace(0.0, plot_vmax, 17)
    levels_line = np.arange(0.0, plot_vmax + 0.5, 0.5)

    mesh = ax.contourf(
        phi_coords,
        psi_coords,
        masked,
        levels=levels_fill,
        cmap="jet",
        vmin=0.0,
        vmax=plot_vmax,
        extend="max",
    )
    if levels_line.size >= 2:
        ax.contour(
            phi_coords,
            psi_coords,
            finite_display.T,
            levels=levels_line,
            colors="black",
            linewidths=0.55,
            alpha=0.65,
        )

    phi_x, phi_y = profile_to_path_xy(phi_coords, psi_coords, phi_path_indices)
    psi_y = psi_coords.astype(float).copy()
    psi_x = np.full(psi_coords.shape, np.nan, dtype=float)
    valid_psi = psi_path_indices >= 0
    psi_x[valid_psi] = phi_coords[psi_path_indices[valid_psi]]

    plot_path_segments(ax, phi_x, phi_y, color="white", linestyle="-", label="phi-MFEP")
    plot_path_segments(ax, psi_x, psi_y, color="black", linestyle="--", label="psi-MFEP")

    ax.scatter(
        [phi_coords[anchor_index[0]]],
        [psi_coords[anchor_index[1]]],
        color="black",
        s=22,
        zorder=5,
        label="global minimum",
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label("PMF (kcal/mol)", fontsize=12)

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_xlabel("phi (deg)", fontsize=13)
    ax.set_ylabel("psi (deg)", fontsize=13)
    ax.set_title("phi/psi 2D PMF", fontsize=14)
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_1d_profiles(
    out_png: Path,
    out_pdf: Path,
    phi_profile: np.ndarray,
    psi_profile: np.ndarray,
    plot_vmax: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True, sharey=True)

    specs = (
        (axes[0], phi_profile, "phi"),
        (axes[1], psi_profile, "psi"),
    )

    for ax, profile, axis_name in specs:
        finite = np.isfinite(profile[:, 2])
        finite_index = np.where(finite)[0]
        if finite_index.size > 0:
            starts = [0]
            starts.extend((np.where(np.diff(finite_index) > 1)[0] + 1).tolist())
            starts.append(finite_index.size)

            for start, end in zip(starts[:-1], starts[1:]):
                segment_index = finite_index[start:end]
                if segment_index.size >= 2:
                    ax.plot(
                        profile[segment_index, 0],
                        profile[segment_index, 2],
                        color="#1f77b4",
                        linewidth=1.8,
                    )

            ax.scatter(
                profile[finite, 0],
                profile[finite, 2],
                s=16,
                color="#1f77b4",
                edgecolors="none",
                zorder=3,
            )

        ax.set_xlim(-180.0, 180.0)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_xlabel(f"{axis_name} (deg)", fontsize=12)
        ax.set_title(f"{axis_name} 1D PMF", fontsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel("PMF (kcal/mol)", fontsize=12)
    axes[0].set_ylim(0.0, plot_vmax)

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build phi/psi 2D PMF from CMD dihedral time-series XVG files and "
            "extract 1D phi/psi PMFs by tracing support-aware MFEPs on the 2D surface."
        )
    )
    parser.add_argument("workdir", help="Directory containing cmd_dih_phi_time.xvg and cmd_dih_psi_time.xvg")
    parser.add_argument("--phi-time", default="cmd_dih_phi_time.xvg", help="Phi time-series XVG, relative to workdir unless absolute")
    parser.add_argument("--psi-time", default="cmd_dih_psi_time.xvg", help="Psi time-series XVG, relative to workdir unless absolute")
    parser.add_argument("--outdir", default="cmd_phi_psi_mfep_open", help="Output directory, relative to workdir unless absolute")
    parser.add_argument("--out-prefix", default="cmd_phi_psi", help="Prefix for output data and figure files")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K for PMF conversion")
    parser.add_argument("--bin-width", type=float, default=1.0, help="Bin width in degrees; must evenly divide 360")
    parser.add_argument("--count-cutoff", type=int, default=1, help="Minimum count retained in the 2D PMF")
    parser.add_argument("--plot-vmax", type=float, default=8.0, help="Maximum PMF shown in plots (kcal/mol)")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma in bins for display only")
    parser.add_argument("--time-tol", type=float, default=1e-6, help="Absolute tolerance for matching phi/psi time grids")
    parser.add_argument("--mfep-count-cutoff", type=int, default=5, help="Minimum count retained when tracing the MFEP")
    parser.add_argument("--mfep-smooth-sigma", type=float, default=1.2, help="Gaussian sigma in bins used only for MFEP tracing")
    parser.add_argument("--mfep-max-step", type=int, default=18, help="Maximum bin shift allowed between adjacent steps along the MFEP")
    parser.add_argument("--mfep-jump-penalty", type=float, default=0.01, help="Quadratic continuity penalty used when tracing the MFEP")
    parser.add_argument("--phi-gap-support-fraction", type=float, default=0.001, help="Phi marginal support threshold, as a fraction of the phi marginal maximum, used to detect a large unsampled gap and switch to open-path tracing")
    parser.add_argument("--psi-gap-support-fraction", type=float, default=0.0, help="Psi marginal support threshold, as a fraction of the psi marginal maximum, used to detect a large unsampled gap and switch to open-path tracing")
    parser.add_argument("--open-gap-min-bins", type=int, default=24, help="Minimum low-support gap size, in bins, required to switch from circular to open-path tracing")
    parser.add_argument("--snap-deviation-penalty", type=float, default=0.02, help="Quadratic penalty used when snapping the traced MFEP back onto sampled raw-PMF bins in the same row")
    parser.add_argument("--snap-jump-penalty", type=float, default=0.02, help="Quadratic continuity penalty used while snapping the MFEP back onto sampled raw-PMF bins")
    parser.add_argument("--phi-support-fraction", type=float, default=0.001, help="Mask phi 1D bins whose marginal count is below this fraction of the phi marginal maximum; set 0 to disable")
    parser.add_argument("--psi-support-fraction", type=float, default=0.0, help="Mask psi 1D bins whose marginal count is below this fraction of the psi marginal maximum; set 0 to disable")
    parser.add_argument("--save-samples", action="store_true", help="Also write the paired phi/psi sample table (can be large)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = workdir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    phi_time_path = Path(args.phi_time)
    psi_time_path = Path(args.psi_time)
    if not phi_time_path.is_absolute():
        phi_time_path = workdir / phi_time_path
    if not psi_time_path.is_absolute():
        psi_time_path = workdir / psi_time_path

    phi_time, phi_values = read_xvg_xy(phi_time_path)
    psi_time, psi_values = read_xvg_xy(psi_time_path)
    ensure_matching_times(phi_time, psi_time, args.time_tol)

    phi_coords, psi_coords, counts, probabilities, pmf = compute_2d_pmf(
        phi_values=phi_values,
        psi_values=psi_values,
        bin_width=args.bin_width,
        temperature=args.temperature,
        count_cutoff=args.count_cutoff,
    )

    anchor_index = np.unravel_index(np.nanargmin(pmf), pmf.shape)
    anchor_index = (int(anchor_index[0]), int(anchor_index[1]))

    phi_profile, phi_path_indices, phi_trace_mode = extract_axis_mfep(
        row_coords=phi_coords,
        col_coords=psi_coords,
        pmf=pmf,
        counts=counts,
        probabilities=probabilities,
        anchor=anchor_index,
        gap_support_fraction=args.phi_gap_support_fraction,
        open_gap_min_bins=args.open_gap_min_bins,
        mfep_count_cutoff=args.mfep_count_cutoff,
        mfep_smooth_sigma=args.mfep_smooth_sigma,
        mfep_max_step=args.mfep_max_step,
        mfep_jump_penalty=args.mfep_jump_penalty,
        snap_deviation_penalty=args.snap_deviation_penalty,
        snap_jump_penalty=args.snap_jump_penalty,
    )
    psi_profile, psi_path_indices, psi_trace_mode = extract_axis_mfep(
        row_coords=psi_coords,
        col_coords=phi_coords,
        pmf=pmf.T,
        counts=counts.T,
        probabilities=probabilities.T,
        anchor=(anchor_index[1], anchor_index[0]),
        gap_support_fraction=args.psi_gap_support_fraction,
        open_gap_min_bins=args.open_gap_min_bins,
        mfep_count_cutoff=args.mfep_count_cutoff,
        mfep_smooth_sigma=args.mfep_smooth_sigma,
        mfep_max_step=args.mfep_max_step,
        mfep_jump_penalty=args.mfep_jump_penalty,
        snap_deviation_penalty=args.snap_deviation_penalty,
        snap_jump_penalty=args.snap_jump_penalty,
    )

    phi_profile, phi_support_threshold = apply_marginal_support_mask(
        profile=phi_profile,
        marginal_counts=np.sum(counts, axis=1),
        support_fraction=args.phi_support_fraction,
    )
    psi_profile, psi_support_threshold = apply_marginal_support_mask(
        profile=psi_profile,
        marginal_counts=np.sum(counts, axis=0),
        support_fraction=args.psi_support_fraction,
    )

    pmf2d_path = outdir / f"{args.out_prefix}_2d_pmf.xvg"
    phi_1d_path = outdir / f"{args.out_prefix}_phi_1d_mfep_pmf.xvg"
    psi_1d_path = outdir / f"{args.out_prefix}_psi_1d_mfep_pmf.xvg"
    pmf2d_png = outdir / f"{args.out_prefix}_2d_pmf.png"
    pmf2d_pdf = outdir / f"{args.out_prefix}_2d_pmf.pdf"
    pmf1d_png = outdir / f"{args.out_prefix}_1d_mfep_pmf.png"
    pmf1d_pdf = outdir / f"{args.out_prefix}_1d_mfep_pmf.pdf"

    samples_path = None
    if args.save_samples:
        samples_path = outdir / f"{args.out_prefix}_samples.xvg"
        write_paired_samples(samples_path, phi_time, phi_values, psi_values)

    write_2d_pmf(pmf2d_path, phi_coords, psi_coords, pmf, counts, probabilities)
    write_1d_profile(phi_1d_path, "phi", "psi", phi_profile)
    write_1d_profile(psi_1d_path, "psi", "phi", psi_profile)
    plot_2d_pmf(
        pmf2d_png,
        pmf2d_pdf,
        phi_coords,
        psi_coords,
        pmf,
        phi_path_indices,
        psi_path_indices,
        anchor_index,
        args.plot_vmax,
        args.smooth_sigma,
    )
    plot_1d_profiles(pmf1d_png, pmf1d_pdf, phi_profile, psi_profile, args.plot_vmax)

    global_min_phi = float(phi_coords[anchor_index[0]])
    global_min_psi = float(psi_coords[anchor_index[1]])

    print("[INFO] Built phi/psi PMFs from paired CMD dihedral trajectories")
    print(f"       frames                  : {phi_time.size}")
    print(f"       time range (ps)         : {phi_time[0]:.5f} -> {phi_time[-1]:.5f}")
    print(f"       bin width (deg)         : {args.bin_width}")
    print(f"       count cutoff            : {args.count_cutoff}")
    print(f"       mfep count cutoff       : {args.mfep_count_cutoff}")
    print(f"       mfep max step           : {args.mfep_max_step}")
    print(f"       phi trace mode          : {phi_trace_mode.mode}")
    print(f"       phi gap threshold       : {phi_trace_mode.support_threshold:.3f}")
    print(f"       phi largest gap (bins)  : start={phi_trace_mode.largest_gap_start}, length={phi_trace_mode.largest_gap_length}")
    print(f"       phi rotation shift      : {phi_trace_mode.rotation_shift}")
    print(f"       psi trace mode          : {psi_trace_mode.mode}")
    print(f"       psi gap threshold       : {psi_trace_mode.support_threshold:.3f}")
    print(f"       psi largest gap (bins)  : start={psi_trace_mode.largest_gap_start}, length={psi_trace_mode.largest_gap_length}")
    print(f"       psi rotation shift      : {psi_trace_mode.rotation_shift}")
    print(f"       phi support frac        : {args.phi_support_fraction}")
    print(f"       psi support frac        : {args.psi_support_fraction}")
    if args.phi_support_fraction > 0.0:
        print(f"       phi support count       : {phi_support_threshold:.3f}")
    if args.psi_support_fraction > 0.0:
        print(f"       psi support count       : {psi_support_threshold:.3f}")
    print(f"       global minimum          : phi={global_min_phi:.3f} deg, psi={global_min_psi:.3f} deg")
    if samples_path is not None:
        print(f"       samples                 : {samples_path}")
    print(f"       2D PMF                 : {pmf2d_path}")
    print(f"       phi 1D PMF             : {phi_1d_path}")
    print(f"       psi 1D PMF             : {psi_1d_path}")
    print(f"       2D figure              : {pmf2d_png}")
    print(f"       2D figure              : {pmf2d_pdf}")
    print(f"       1D figure              : {pmf1d_png}")
    print(f"       1D figure              : {pmf1d_pdf}")


if __name__ == "__main__":
    main()

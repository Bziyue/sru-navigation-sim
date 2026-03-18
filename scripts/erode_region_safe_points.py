#!/usr/bin/env python3
"""Erode precomputed region safe points by expanding empty grid cells outward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _build_uniform_axis(low: float, high: float, spacing: float) -> np.ndarray:
    span = float(high - low)
    if span <= spacing:
        return np.asarray([(low + high) * 0.5], dtype=np.float32)

    axis = np.arange(low + 0.5 * spacing, high, spacing, dtype=np.float32)
    if len(axis) == 0:
        axis = np.asarray([(low + high) * 0.5], dtype=np.float32)
    return axis.astype(np.float32, copy=False)


def _neighbor_offsets(radius_m: float, xy_spacing: float, z_spacing: float) -> list[tuple[int, int, int]]:
    dx_max = int(np.floor(radius_m / xy_spacing + 1e-6))
    dy_max = int(np.floor(radius_m / xy_spacing + 1e-6))
    dz_max = int(np.floor(radius_m / z_spacing + 1e-6))

    offsets: list[tuple[int, int, int]] = []
    for dy in range(-dy_max, dy_max + 1):
        for dx in range(-dx_max, dx_max + 1):
            for dz in range(-dz_max, dz_max + 1):
                distance = float(np.sqrt((dx * xy_spacing) ** 2 + (dy * xy_spacing) ** 2 + (dz * z_spacing) ** 2))
                if distance <= radius_m + 1e-6:
                    offsets.append((dy, dx, dz))
    return offsets


def _shift_or(mask: np.ndarray, offset: tuple[int, int, int], out: np.ndarray) -> None:
    dy, dx, dz = offset
    src_y_start = max(0, -dy)
    src_y_end = mask.shape[0] - max(0, dy)
    src_x_start = max(0, -dx)
    src_x_end = mask.shape[1] - max(0, dx)
    src_z_start = max(0, -dz)
    src_z_end = mask.shape[2] - max(0, dz)

    dst_y_start = max(0, dy)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, dx)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_z_start = max(0, dz)
    dst_z_end = dst_z_start + (src_z_end - src_z_start)

    if src_y_start >= src_y_end or src_x_start >= src_x_end or src_z_start >= src_z_end:
        return

    out[dst_y_start:dst_y_end, dst_x_start:dst_x_end, dst_z_start:dst_z_end] |= mask[
        src_y_start:src_y_end,
        src_x_start:src_x_end,
        src_z_start:src_z_end,
    ]


def _region_safe_points_to_mask(
    region_points: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_values: np.ndarray,
) -> np.ndarray:
    mask = np.zeros((len(y_axis), len(x_axis), len(z_values)), dtype=bool)
    if len(region_points) == 0:
        return mask

    x_indices = np.rint((region_points[:, 0] - x_axis[0]) / (x_axis[1] - x_axis[0] if len(x_axis) > 1 else 1.0)).astype(np.int64)
    y_indices = np.rint((region_points[:, 1] - y_axis[0]) / (y_axis[1] - y_axis[0] if len(y_axis) > 1 else 1.0)).astype(np.int64)
    z_indices = np.rint((region_points[:, 2] - z_values[0]) / (z_values[1] - z_values[0] if len(z_values) > 1 else 1.0)).astype(np.int64)
    mask[y_indices, x_indices, z_indices] = True
    return mask


def _mask_to_points(mask: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_values: np.ndarray) -> np.ndarray:
    y_indices, x_indices, z_indices = np.where(mask)
    if len(y_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack(
        (
            x_axis[x_indices],
            y_axis[y_indices],
            z_values[z_indices],
        )
    ).astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand empty safe-point grid cells and remove nearby safe points.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/zdp/CodeField/my_swarm_rl/rl/sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Environments/StaticScan/DR_region_safe_points_contact_0p2m_1p2_to_2p0.npz"),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.4,
        help="Expansion radius in meters around empty cells.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zdp/CodeField/my_swarm_rl/rl/sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Environments/StaticScan/DR_region_safe_points_contact_0p2m_1p2_to_2p0_eroded_0p4m.npz"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/home/zdp/CodeField/my_swarm_rl/rl/sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Environments/StaticScan/DR_region_safe_points_contact_0p2m_1p2_to_2p0_eroded_0p4m.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = np.load(args.input)

    region_names = np.asarray(payload["region_names"]).astype(str)
    region_xy_min = np.asarray(payload["region_xy_min"], dtype=np.float32)
    region_xy_max = np.asarray(payload["region_xy_max"], dtype=np.float32)
    region_center_xy = np.asarray(payload["region_center_xy"], dtype=np.float32)
    region_floor_z = np.asarray(payload["region_floor_z"], dtype=np.float32)
    region_start_indices = np.asarray(payload["region_start_indices"], dtype=np.int64)
    region_counts = np.asarray(payload["region_counts"], dtype=np.int64)
    region_candidate_counts = np.asarray(payload["region_candidate_counts"], dtype=np.int64)
    points_xyz = np.asarray(payload["points_xyz"], dtype=np.float32)
    z_values = np.asarray(payload["z_values"], dtype=np.float32)
    xy_spacing = float(np.asarray(payload["xy_spacing"], dtype=np.float32).reshape(-1)[0])
    z_spacing = float(np.asarray(payload["z_spacing"], dtype=np.float32).reshape(-1)[0])

    offsets = _neighbor_offsets(radius_m=float(args.radius), xy_spacing=xy_spacing, z_spacing=z_spacing)

    new_region_start_indices: list[int] = []
    new_region_counts: list[int] = []
    new_region_safe_ratios: list[float] = []
    all_points_new: list[np.ndarray] = []
    summary_regions: list[dict[str, object]] = []
    running_start = 0

    for region_id in range(len(region_names)):
        start = int(region_start_indices[region_id])
        count = int(region_counts[region_id])
        region_points = points_xyz[start : start + count]
        x_axis = _build_uniform_axis(float(region_xy_min[region_id, 0]), float(region_xy_max[region_id, 0]), xy_spacing)
        y_axis = _build_uniform_axis(float(region_xy_min[region_id, 1]), float(region_xy_max[region_id, 1]), xy_spacing)

        safe_mask = _region_safe_points_to_mask(region_points=region_points, x_axis=x_axis, y_axis=y_axis, z_values=z_values)
        unsafe_mask = ~safe_mask
        expanded_unsafe_mask = np.zeros_like(unsafe_mask)
        for offset in offsets:
            _shift_or(unsafe_mask, offset, expanded_unsafe_mask)

        eroded_safe_mask = safe_mask & ~expanded_unsafe_mask
        eroded_points = _mask_to_points(eroded_safe_mask, x_axis=x_axis, y_axis=y_axis, z_values=z_values)

        new_region_start_indices.append(running_start)
        new_region_counts.append(int(len(eroded_points)))
        safe_ratio = 0.0 if int(region_candidate_counts[region_id]) == 0 else float(len(eroded_points) / int(region_candidate_counts[region_id]))
        new_region_safe_ratios.append(safe_ratio)
        all_points_new.append(eroded_points)
        running_start += int(len(eroded_points))

        summary_regions.append(
            {
                "region_id": region_id,
                "name": str(region_names[region_id]),
                "candidate_count": int(region_candidate_counts[region_id]),
                "safe_count_before": int(count),
                "safe_count_after": int(len(eroded_points)),
                "removed_count": int(count - len(eroded_points)),
                "safe_ratio_after": safe_ratio,
                "xy_min": [float(v) for v in region_xy_min[region_id]],
                "xy_max": [float(v) for v in region_xy_max[region_id]],
                "floor_z": float(region_floor_z[region_id]),
            }
        )

    points_xyz_new = (
        np.concatenate(all_points_new, axis=0).astype(np.float32, copy=False)
        if all_points_new
        else np.zeros((0, 3), dtype=np.float32)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        region_names=region_names.astype(np.str_),
        region_xy_min=region_xy_min.astype(np.float32, copy=False),
        region_xy_max=region_xy_max.astype(np.float32, copy=False),
        region_center_xy=region_center_xy.astype(np.float32, copy=False),
        region_floor_z=region_floor_z.astype(np.float32, copy=False),
        region_start_indices=np.asarray(new_region_start_indices, dtype=np.int64),
        region_counts=np.asarray(new_region_counts, dtype=np.int64),
        region_candidate_counts=region_candidate_counts.astype(np.int64, copy=False),
        region_safe_ratios=np.asarray(new_region_safe_ratios, dtype=np.float32),
        points_xyz=points_xyz_new,
        z_values=z_values.astype(np.float32, copy=False),
        xy_spacing=np.asarray([xy_spacing], dtype=np.float32),
        z_spacing=np.asarray([z_spacing], dtype=np.float32),
        z_min=np.asarray(payload["z_min"], dtype=np.float32),
        z_max=np.asarray(payload["z_max"], dtype=np.float32),
        contact_threshold=np.asarray(payload["contact_threshold"], dtype=np.float32),
        collision_mesh_path=np.asarray(payload["collision_mesh_path"]).astype(np.str_),
        surface_bbox_data_path=np.asarray(payload["surface_bbox_data_path"]).astype(np.str_),
        task=np.asarray(payload["task"]).astype(np.str_),
    )

    total_before = int(np.sum(region_counts))
    total_after = int(np.sum(new_region_counts))
    total_candidates = int(np.sum(region_candidate_counts))
    summary = {
        "input_npz": str(args.input),
        "output_npz": str(args.output),
        "radius_m": float(args.radius),
        "xy_spacing": xy_spacing,
        "z_spacing": z_spacing,
        "z_values": [float(v) for v in z_values],
        "total_regions": int(len(region_names)),
        "total_candidate_points": total_candidates,
        "total_safe_points_before": total_before,
        "total_safe_points_after": total_after,
        "removed_points": int(total_before - total_after),
        "overall_safe_ratio_after": 0.0 if total_candidates == 0 else float(total_after / total_candidates),
        "regions": summary_regions,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved eroded safe points to {args.output}")
    print(f"[INFO] Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()

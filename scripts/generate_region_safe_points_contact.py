#!/usr/bin/env python3
"""Generate region-wise safe point pools using the same contact-force logic as training."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


def _prepend_workspace_packages() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_root = os.path.dirname(script_dir)
    workspace_root = os.path.dirname(sim_root)
    learning_root = os.path.join(workspace_root, "sru-navigation-learning")

    for path in [learning_root, sim_root]:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_prepend_workspace_packages()

parser = argparse.ArgumentParser(description="Generate safe region points with Isaac Sim contact checks.")
parser.add_argument("--task", type=str, default="Isaac-Nav-PPO-Drone-Static-PlayFast-v0")
parser.add_argument("--xy-spacing", type=float, default=0.2, help="XY grid spacing inside each region.")
parser.add_argument("--z-min", type=float, default=1.2, help="Minimum candidate z height.")
parser.add_argument("--z-max", type=float, default=2.0, help="Maximum candidate z height.")
parser.add_argument("--z-spacing", type=float, default=0.2, help="Height sampling step.")
parser.add_argument(
    "--contact-threshold",
    type=float,
    default=-1.0,
    help="Collision threshold on contact-force norm. Negative means read from task config.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=1,
    help="Number of PhysX steps to run while pinning the drone at each candidate point.",
)
parser.add_argument("--max-regions", type=int, default=0, help="Only process the first N regions. 0 means all regions.")
parser.add_argument(
    "--output",
    type=str,
    default="/home/zdp/CodeField/my_swarm_rl/rl/sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Environments/StaticScan/DR_region_safe_points_contact_0p2m_1p2_to_2p0.npz",
    help="Compressed NPZ file storing safe points grouped by region.",
)
parser.add_argument(
    "--summary-json",
    type=str,
    default="",
    help="Optional JSON summary path. Defaults next to --output with the same stem.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
import isaaclab_nav_task.navigation.config.drone  # noqa: F401

from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


_RECT_NAME_RE = re.compile(r"^Rectangle:\s*(.+)$")
_CORNER_RE = re.compile(r"Corner\s+\d+:\s+X:\s*([-+0-9.]+),\s*Y:\s*([-+0-9.]+),\s*Z:\s*([-+0-9.]+)")


@dataclass(frozen=True)
class RegionBox:
    region_id: int
    name: str
    xy_min: np.ndarray
    xy_max: np.ndarray
    center_xy: np.ndarray
    floor_z: float


def _load_region_boxes(path: str) -> list[RegionBox]:
    regions: list[RegionBox] = []
    current_name: str | None = None
    current_points: list[list[float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            match = _RECT_NAME_RE.match(line)
            if match is not None:
                if current_name is not None and current_points:
                    points = np.asarray(current_points, dtype=np.float32)
                    xy_min = points[:, :2].min(axis=0).astype(np.float32, copy=False)
                    xy_max = points[:, :2].max(axis=0).astype(np.float32, copy=False)
                    regions.append(
                        RegionBox(
                            region_id=len(regions),
                            name=current_name,
                            xy_min=xy_min,
                            xy_max=xy_max,
                            center_xy=((xy_min + xy_max) * 0.5).astype(np.float32, copy=False),
                            floor_z=float(points[:, 2].min()),
                        )
                    )
                current_name = match.group(1)
                current_points = []
                continue

            match = _CORNER_RE.search(line)
            if match is not None:
                current_points.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])

    if current_name is not None and current_points:
        points = np.asarray(current_points, dtype=np.float32)
        xy_min = points[:, :2].min(axis=0).astype(np.float32, copy=False)
        xy_max = points[:, :2].max(axis=0).astype(np.float32, copy=False)
        regions.append(
            RegionBox(
                region_id=len(regions),
                name=current_name,
                xy_min=xy_min,
                xy_max=xy_max,
                center_xy=((xy_min + xy_max) * 0.5).astype(np.float32, copy=False),
                floor_z=float(points[:, 2].min()),
            )
        )

    if not regions:
        raise RuntimeError(f"No region boxes found in {path}")
    return regions


def _build_uniform_axis(low: float, high: float, spacing: float) -> np.ndarray:
    span = float(high - low)
    if span <= spacing:
        return np.asarray([(low + high) * 0.5], dtype=np.float32)

    axis = np.arange(low + 0.5 * spacing, high, spacing, dtype=np.float32)
    if len(axis) == 0:
        axis = np.asarray([(low + high) * 0.5], dtype=np.float32)
    return axis.astype(np.float32, copy=False)


def _build_height_axis(z_min: float, z_max: float, z_spacing: float) -> np.ndarray:
    if z_max < z_min:
        raise ValueError(f"Expected z_max >= z_min, got {z_min} -> {z_max}")
    if z_spacing <= 0.0:
        raise ValueError(f"Expected z_spacing > 0, got {z_spacing}")

    axis = np.arange(z_min, z_max + 0.5 * z_spacing, z_spacing, dtype=np.float32)
    if len(axis) == 0:
        axis = np.asarray([z_min], dtype=np.float32)
    axis[-1] = np.float32(z_max)
    return np.unique(np.round(axis, decimals=4)).astype(np.float32, copy=False)


def _build_candidate_points(region: RegionBox, xy_spacing: float, z_values: np.ndarray) -> np.ndarray:
    x_axis = _build_uniform_axis(float(region.xy_min[0]), float(region.xy_max[0]), xy_spacing)
    y_axis = _build_uniform_axis(float(region.xy_min[1]), float(region.xy_max[1]), xy_spacing)
    grid_x, grid_y, grid_z = np.meshgrid(x_axis, y_axis, z_values, indexing="xy")
    return np.column_stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            grid_z.reshape(-1),
        )
    ).astype(np.float32, copy=False)


def _get_contact_threshold(env_cfg) -> float:
    threshold = float(env_cfg.terminations.body_contact.params["threshold"])
    if args_cli.contact_threshold >= 0.0:
        threshold = float(args_cli.contact_threshold)
    return threshold


def _step_sim_once(base_env) -> None:
    base_env.scene.write_data_to_sim()
    base_env.sim.step(render=False)
    base_env.scene.update(dt=base_env.physics_dt)


def _pin_robot_at_pose(base_env, robot, position_xyz: np.ndarray, yaw_rad: float) -> None:
    position = torch.tensor(position_xyz, device=base_env.device, dtype=torch.float32).unsqueeze(0)
    zero = torch.zeros(1, device=base_env.device, dtype=torch.float32)
    quat = quat_from_euler_xyz(zero, zero, torch.tensor([yaw_rad], device=base_env.device, dtype=torch.float32))
    root_pose = torch.cat([position, quat], dim=1)
    root_velocity = torch.zeros((1, 6), device=base_env.device, dtype=torch.float32)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)


def _is_collision_free(base_env, robot, contact_sensor, body_id: int, point_xyz: np.ndarray, contact_threshold: float) -> tuple[bool, float]:
    contact_sensor.reset()
    for _ in range(max(int(args_cli.settle_steps), 1)):
        _pin_robot_at_pose(base_env, robot, point_xyz, yaw_rad=0.0)
        _step_sim_once(base_env)

    net_forces_hist = contact_sensor.data.net_forces_w_history[0, :, body_id]
    hist_force_norms = torch.norm(net_forces_hist, dim=-1)
    history_max_contact_force_norm = float(torch.max(hist_force_norms).item())
    return history_max_contact_force_norm <= contact_threshold, history_max_contact_force_norm


def _default_summary_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def main() -> None:
    env = None
    try:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        env_cfg.scene.num_envs = 1

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env.reset()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        contact_sensor = base_env.scene.sensors["contact_forces"]
        body_ids, body_names = contact_sensor.find_bodies(["body"], preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected exactly one contact body named 'body', got: {body_names}")
        body_id = int(body_ids[0])
        command_term = base_env.command_manager._terms["robot_goal"]
        regions = _load_region_boxes(command_term.cfg.surface_bbox_data_path)
        z_values = _build_height_axis(args_cli.z_min, args_cli.z_max, args_cli.z_spacing)
        contact_threshold = _get_contact_threshold(env_cfg)
        output_path = Path(args_cli.output)
        summary_path = Path(args_cli.summary_json) if args_cli.summary_json else _default_summary_path(output_path)

        processed_region_limit = args_cli.max_regions if args_cli.max_regions > 0 else len(regions)
        all_safe_points: list[np.ndarray] = []
        region_start_indices: list[int] = []
        region_counts: list[int] = []
        region_candidate_counts: list[int] = []
        region_safe_ratios: list[float] = []
        summary_regions: list[dict[str, object]] = []

        print("[INFO] task:", args_cli.task, flush=True)
        print("[INFO] contact_threshold:", contact_threshold, flush=True)
        print("[INFO] z_values:", [float(v) for v in z_values], flush=True)
        print("[INFO] regions:", processed_region_limit, "/", len(regions), flush=True)

        running_start = 0
        for region in regions:
            if region.region_id >= processed_region_limit:
                all_safe_points.append(np.zeros((0, 3), dtype=np.float32))
                region_start_indices.append(running_start)
                region_counts.append(0)
                region_candidate_counts.append(0)
                region_safe_ratios.append(0.0)
                summary_regions.append(
                    {
                        "region_id": region.region_id,
                        "name": region.name,
                        "candidate_count": 0,
                        "safe_count": 0,
                        "safe_ratio": 0.0,
                        "max_history_contact_force_norm": 0.0,
                        "xy_min": [float(v) for v in region.xy_min],
                        "xy_max": [float(v) for v in region.xy_max],
                        "floor_z": float(region.floor_z),
                        "processed": False,
                    }
                )
                continue

            candidates = _build_candidate_points(region, xy_spacing=args_cli.xy_spacing, z_values=z_values)
            safe_points_region: list[np.ndarray] = []
            max_force = 0.0

            print(
                f"[REGION] {region.region_id:02d} {region.name} "
                f"candidates={len(candidates)} xy=[{region.xy_min.tolist()} -> {region.xy_max.tolist()}]",
                flush=True,
            )

            for index, candidate in enumerate(candidates):
                is_safe, history_force = _is_collision_free(
                    base_env=base_env,
                    robot=robot,
                    contact_sensor=contact_sensor,
                    body_id=body_id,
                    point_xyz=candidate,
                    contact_threshold=contact_threshold,
                )
                max_force = max(max_force, history_force)
                if is_safe:
                    safe_points_region.append(candidate)

                if (index + 1) % 500 == 0 or index + 1 == len(candidates):
                    print(
                        f"  checked={index + 1}/{len(candidates)} safe={len(safe_points_region)} "
                        f"maxF={max_force:.6f}",
                        flush=True,
                    )

            safe_points_np = (
                np.asarray(safe_points_region, dtype=np.float32)
                if safe_points_region
                else np.zeros((0, 3), dtype=np.float32)
            )
            all_safe_points.append(safe_points_np)
            region_start_indices.append(running_start)
            region_counts.append(int(len(safe_points_np)))
            region_candidate_counts.append(int(len(candidates)))
            safe_ratio = 0.0 if len(candidates) == 0 else float(len(safe_points_np) / len(candidates))
            region_safe_ratios.append(safe_ratio)
            running_start += int(len(safe_points_np))

            summary_regions.append(
                {
                    "region_id": region.region_id,
                    "name": region.name,
                    "candidate_count": int(len(candidates)),
                    "safe_count": int(len(safe_points_np)),
                    "safe_ratio": safe_ratio,
                    "max_history_contact_force_norm": float(max_force),
                    "xy_min": [float(v) for v in region.xy_min],
                    "xy_max": [float(v) for v in region.xy_max],
                    "floor_z": float(region.floor_z),
                    "processed": True,
                }
            )

        points_xyz = (
            np.concatenate(all_safe_points, axis=0).astype(np.float32, copy=False)
            if all_safe_points
            else np.zeros((0, 3), dtype=np.float32)
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            region_names=np.asarray([region.name for region in regions], dtype=np.str_),
            region_xy_min=np.asarray([region.xy_min for region in regions], dtype=np.float32),
            region_xy_max=np.asarray([region.xy_max for region in regions], dtype=np.float32),
            region_center_xy=np.asarray([region.center_xy for region in regions], dtype=np.float32),
            region_floor_z=np.asarray([region.floor_z for region in regions], dtype=np.float32),
            region_start_indices=np.asarray(region_start_indices, dtype=np.int64),
            region_counts=np.asarray(region_counts, dtype=np.int64),
            region_candidate_counts=np.asarray(region_candidate_counts, dtype=np.int64),
            region_safe_ratios=np.asarray(region_safe_ratios, dtype=np.float32),
            points_xyz=points_xyz,
            z_values=z_values.astype(np.float32, copy=False),
            xy_spacing=np.asarray([args_cli.xy_spacing], dtype=np.float32),
            z_spacing=np.asarray([args_cli.z_spacing], dtype=np.float32),
            z_min=np.asarray([args_cli.z_min], dtype=np.float32),
            z_max=np.asarray([args_cli.z_max], dtype=np.float32),
            contact_threshold=np.asarray([contact_threshold], dtype=np.float32),
            collision_mesh_path=np.asarray([command_term.cfg.map_mesh_prim_path], dtype=np.str_),
            surface_bbox_data_path=np.asarray([command_term.cfg.surface_bbox_data_path], dtype=np.str_),
            task=np.asarray([args_cli.task], dtype=np.str_),
        )

        summary = {
            "task": args_cli.task,
            "output_npz": str(output_path),
            "surface_bbox_data_path": command_term.cfg.surface_bbox_data_path,
            "collision_mesh_path": command_term.cfg.map_mesh_prim_path,
            "collision_approximation": "sdf",
            "xy_spacing": float(args_cli.xy_spacing),
            "z_values": [float(v) for v in z_values],
            "settle_steps": int(args_cli.settle_steps),
            "contact_threshold": float(contact_threshold),
            "total_regions": len(regions),
            "processed_regions": int(processed_region_limit),
            "total_safe_points": int(len(points_xyz)),
            "total_candidate_points": int(sum(region_candidate_counts)),
            "overall_safe_ratio": 0.0
            if sum(region_candidate_counts) == 0
            else float(len(points_xyz) / sum(region_candidate_counts)),
            "regions": summary_regions,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("[SUMMARY]", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print("[INFO] Safe-point pool saved to", output_path, flush=True)
        print("[INFO] Summary saved to", summary_path, flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

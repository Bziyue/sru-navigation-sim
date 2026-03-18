#!/usr/bin/env python3
"""Validate fixed-height reachability for the static drone navigation scene.

This script launches the Isaac Lab environment used by training, reuses the same
merged collision mesh processing, teleports the real drone articulation to every
path sample at a fixed height, advances PhysX, and reads the configured
``contact_forces`` sensor to determine collisions exactly as the training setup does.

The path processing is intentionally unchanged:
- ``trajectory_xy``: sample the optimized quintic trajectory, then project it to a fixed z
- ``resampled_path``: use the exported path points, then project them to a fixed z

For later visualization, each directed region-pair gets a compressed ``.npz`` file
containing every checked point together with contact-force statistics and collision flags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def _prepend_workspace_packages() -> None:
    """Prefer the local workspace packages over stale site-packages installs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_root = os.path.dirname(script_dir)
    workspace_root = os.path.dirname(sim_root)
    learning_root = os.path.join(workspace_root, "sru-navigation-learning")

    for path in [learning_root, sim_root]:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_prepend_workspace_packages()

parser = argparse.ArgumentParser(description="Validate fixed-height reachability for static drone guidance paths.")
parser.add_argument("--task", type=str, default="Isaac-Nav-PPO-Drone-Static-PlayFast-v0", help="Task name.")
parser.add_argument("--flight_height", type=float, default=1.2, help="Fixed drone center height in meters.")
parser.add_argument(
    "--path_source",
    type=str,
    choices=("trajectory_xy", "resampled_path"),
    default="trajectory_xy",
    help="Path representation to validate at fixed height.",
)
parser.add_argument(
    "--path_spacing",
    type=float,
    default=0.05,
    help="Arc-length spacing used to resample each path before collision checking.",
)
parser.add_argument(
    "--trajectory_eval_dt",
    type=float,
    default=0.05,
    help="Sampling dt for quintic trajectories before arc-length resampling.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=1,
    help="Number of PhysX steps to run while pinning the drone at each sampled path point.",
)
parser.add_argument(
    "--contact_threshold",
    type=float,
    default=-1.0,
    help="Collision threshold on contact-force norm. Negative means read it from the task config.",
)
parser.add_argument(
    "--max_pairs",
    type=int,
    default=0,
    help="Validate only the first N directed pairs. 0 means all pairs.",
)
parser.add_argument(
    "--save_point_details",
    action="store_true",
    default=True,
    help="Save per-point collision status for each trajectory as compressed NPZ.",
)
parser.add_argument(
    "--points_dir",
    type=str,
    default="",
    help="Optional directory for per-trajectory point detail files. Defaults next to --output.",
)
parser.add_argument(
    "--output",
    type=str,
    default="logs/fixed_height_validation/fixed_height_validation_report.json",
    help="Path to the JSON report.",
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

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Usd, UsdGeom


@dataclass(slots=True)
class DroneEnvelope:
    prim_path: str
    root_path: str
    xy_radius: float
    z_min_rel: float
    z_max_rel: float
    bounding_sphere_radius: float
    bbox_min_world: tuple[float, float, float]
    bbox_max_world: tuple[float, float, float]


def _compute_prim_world_bbox(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bbox_min = bbox.GetMin()
    bbox_max = bbox.GetMax()
    return (
        np.asarray([float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])], dtype=np.float64),
        np.asarray([float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])], dtype=np.float64),
    )


def _extract_drone_envelope(env) -> DroneEnvelope:
    stage = sim_utils.get_current_stage()
    root_path = f"{env.scene.env_prim_paths[0]}/Robot"
    body_path = f"{root_path}/body"
    candidate_path = body_path if stage.GetPrimAtPath(body_path).IsValid() else root_path

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim = stage.GetPrimAtPath(candidate_path)
    prim_transform = xform_cache.GetLocalToWorldTransform(prim)
    prim_translation = prim_transform.ExtractTranslation()

    bbox_min, bbox_max = _compute_prim_world_bbox(stage, candidate_path)
    prim_origin = np.asarray(
        [float(prim_translation[0]), float(prim_translation[1]), float(prim_translation[2])],
        dtype=np.float64,
    )

    xy_radius = 0.0
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            xy_radius = max(xy_radius, math.hypot(x - float(prim_origin[0]), y - float(prim_origin[1])))

    z_min_rel = float(bbox_min[2] - prim_origin[2])
    z_max_rel = float(bbox_max[2] - prim_origin[2])

    sphere_radius = 0.0
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            for z in [bbox_min[2], bbox_max[2]]:
                sphere_radius = max(
                    sphere_radius,
                    math.sqrt((x - prim_origin[0]) ** 2 + (y - prim_origin[1]) ** 2 + (z - prim_origin[2]) ** 2),
                )

    margin = 0.0
    return DroneEnvelope(
        prim_path=candidate_path,
        root_path=root_path,
        xy_radius=float(xy_radius + margin),
        z_min_rel=float(z_min_rel - margin),
        z_max_rel=float(z_max_rel + margin),
        bounding_sphere_radius=float(sphere_radius + margin),
        bbox_min_world=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max_world=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
    )

def _resample_polyline(points_xyz: np.ndarray, spacing: float, fixed_height: float) -> np.ndarray:
    if len(points_xyz) < 2:
        raise ValueError("Need at least two points to resample a path.")

    deltas = np.diff(points_xyz[:, :2], axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)], axis=0)
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        raise ValueError("Path length is too small.")

    sample_arcs = np.arange(0.0, total_length, spacing, dtype=np.float32)
    if len(sample_arcs) == 0 or not np.isclose(sample_arcs[-1], total_length):
        sample_arcs = np.concatenate([sample_arcs, np.asarray([total_length], dtype=np.float32)])

    resampled = np.column_stack(
        [
            np.interp(sample_arcs, cumulative, points_xyz[:, axis]).astype(np.float32, copy=False)
            for axis in range(3)
        ]
    ).astype(np.float32, copy=False)
    resampled[:, 2] = fixed_height
    return resampled


def _sample_quintic_trajectory_points(trajectory_record: dict[str, Any], eval_dt: float) -> np.ndarray:
    breakpoints = trajectory_record["breakpoints"]
    coefficients = np.asarray(trajectory_record["coefficients"], dtype=np.float32)
    num_segments = len(breakpoints) - 1
    coeffs = coefficients.reshape(num_segments, 6, 3)

    sampled_points: list[np.ndarray] = []
    for segment_idx in range(num_segments):
        duration = float(breakpoints[segment_idx + 1] - breakpoints[segment_idx])
        if duration <= 0.0:
            continue

        local_times = np.arange(0.0, duration, eval_dt, dtype=np.float32)
        if len(local_times) == 0 or not np.isclose(local_times[-1], duration):
            local_times = np.concatenate([local_times, np.asarray([duration], dtype=np.float32)])

        powers = np.stack([np.power(local_times, power, dtype=np.float32) for power in range(6)], axis=1)
        segment_points = np.einsum("tk,kc->tc", powers, coeffs[segment_idx]).astype(np.float32, copy=False)
        sampled_points.append(segment_points)

    if not sampled_points:
        raise ValueError("Failed to sample any quintic trajectory points.")

    dense = np.concatenate(sampled_points, axis=0)
    _, unique_indices = np.unique(np.round(dense[:, :2], decimals=4), axis=0, return_index=True)
    return dense[np.sort(unique_indices)]


def _build_fixed_height_path(
    trajectory_record: dict[str, Any],
    path_source: str,
    path_spacing: float,
    fixed_height: float,
    trajectory_eval_dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if path_source == "resampled_path":
        raw = np.asarray(trajectory_record["resampled_path"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("Invalid resampled_path shape.")
        z_span = float(np.max(raw[:, 2]) - np.min(raw[:, 2]))
        z_mean = float(np.mean(raw[:, 2]))
        return _resample_polyline(raw, path_spacing, fixed_height), {"original_z_span": z_span, "original_z_mean": z_mean}

    if path_source == "trajectory_xy":
        dense = _sample_quintic_trajectory_points(trajectory_record, eval_dt=trajectory_eval_dt)
        z_span = float(np.max(dense[:, 2]) - np.min(dense[:, 2]))
        z_mean = float(np.mean(dense[:, 2]))
        return _resample_polyline(dense, path_spacing, fixed_height), {"original_z_span": z_span, "original_z_mean": z_mean}

    raise ValueError(f"Unsupported path source: {path_source}")


def _compute_path_yaws(path_xyz: np.ndarray) -> np.ndarray:
    forward = np.diff(path_xyz[:, :2], axis=0)
    if len(forward) == 0:
        return np.zeros(1, dtype=np.float32)
    yaws = np.arctan2(forward[:, 1], forward[:, 0]).astype(np.float32)
    return np.concatenate([yaws, yaws[-1:]], axis=0)


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


def _evaluate_path_with_contact_sensor(
    base_env,
    path_xyz: np.ndarray,
    contact_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    yaws = _compute_path_yaws(path_xyz)
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    body_ids, body_names = contact_sensor.find_bodies(["body"], preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected exactly one contact body named 'body', got: {body_names}")
    body_id = int(body_ids[0])

    latest_force_norms = np.zeros(len(path_xyz), dtype=np.float32)
    history_max_force_norms = np.zeros(len(path_xyz), dtype=np.float32)
    latest_force_vectors = np.zeros((len(path_xyz), 3), dtype=np.float32)
    collided_latest = np.zeros(len(path_xyz), dtype=bool)
    collided_history = np.zeros(len(path_xyz), dtype=bool)
    first_collision: dict[str, Any] | None = None

    for index, point in enumerate(path_xyz):
        contact_sensor.reset()
        for _ in range(max(int(args_cli.settle_steps), 1)):
            _pin_robot_at_pose(base_env, robot, point, float(yaws[index]))
            _step_sim_once(base_env)

        net_forces_hist = contact_sensor.data.net_forces_w_history[0, :, body_id]
        latest_force = net_forces_hist[0]
        latest_force_norm = float(torch.norm(latest_force, dim=-1).item())
        hist_force_norms = torch.norm(net_forces_hist, dim=-1)
        history_max_force_norm = float(torch.max(hist_force_norms).item())

        latest_force_norms[index] = latest_force_norm
        history_max_force_norms[index] = history_max_force_norm
        latest_force_vectors[index] = latest_force.detach().cpu().numpy().astype(np.float32, copy=False)
        collided_latest[index] = latest_force_norm > contact_threshold
        collided_history[index] = history_max_force_norm > contact_threshold

        if collided_history[index] and first_collision is None:
            first_collision = {
                "path_index": index,
                "path_point": [float(point[0]), float(point[1]), float(point[2])],
                "yaw_rad": float(yaws[index]),
                "latest_contact_force_norm": latest_force_norm,
                "history_max_contact_force_norm": history_max_force_norm,
                "latest_contact_force_w": [float(v) for v in latest_force_vectors[index]],
            }

    pair_stats = {
        "collision_free": not bool(np.any(collided_history)),
        "first_collision": first_collision,
        "max_latest_contact_force_norm": float(np.max(latest_force_norms)) if len(latest_force_norms) > 0 else 0.0,
        "max_history_contact_force_norm": float(np.max(history_max_force_norms)) if len(history_max_force_norms) > 0 else 0.0,
        "num_collided_points_latest": int(np.count_nonzero(collided_latest)),
        "num_collided_points_history": int(np.count_nonzero(collided_history)),
    }
    point_stats = {
        "path_xyz": path_xyz.astype(np.float32, copy=False),
        "yaw_rad": yaws.astype(np.float32, copy=False),
        "latest_contact_force_norm": latest_force_norms,
        "history_max_contact_force_norm": history_max_force_norms,
        "latest_contact_force_w": latest_force_vectors,
        "collided_latest": collided_latest,
        "collided_history": collided_history,
    }
    return pair_stats, point_stats


def _load_trajectory_records(env) -> tuple[list[dict[str, Any]], str]:
    command_term = env.command_manager._terms["robot_goal"]
    trajectory_path = Path(command_term.cfg.guidance_trajectories_data_path)
    with trajectory_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    trajectories = payload.get("trajectories", [])
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(f"No trajectories found in {trajectory_path}")
    return trajectories, str(trajectory_path)


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    collision_free = sum(1 for item in results if item["collision_free"])
    collided = total - collision_free

    max_history_forces = [item["max_history_contact_force_norm"] for item in results]
    collided_point_counts = [item["num_collided_points_history"] for item in results]

    return {
        "total_pairs_checked": total,
        "collision_free_pairs": collision_free,
        "collided_pairs": collided,
        "collision_free_ratio": 0.0 if total == 0 else float(collision_free / total),
        "collision_ratio": 0.0 if total == 0 else float(collided / total),
        "max_history_contact_force_norm_max": 0.0 if not max_history_forces else float(max(max_history_forces)),
        "max_history_contact_force_norm_mean": 0.0
        if not max_history_forces
        else float(sum(max_history_forces) / len(max_history_forces)),
        "collided_point_count_mean": 0.0 if not collided_point_counts else float(sum(collided_point_counts) / len(collided_point_counts)),
    }


def _default_points_dir(output_path: Path) -> Path:
    if args_cli.points_dir:
        return Path(args_cli.points_dir)
    return output_path.parent / f"{output_path.stem}_points"


def _save_point_details(detail_path: Path, point_stats: dict[str, np.ndarray], contact_threshold: float) -> None:
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        detail_path,
        path_xyz=point_stats["path_xyz"],
        yaw_rad=point_stats["yaw_rad"],
        latest_contact_force_norm=point_stats["latest_contact_force_norm"],
        history_max_contact_force_norm=point_stats["history_max_contact_force_norm"],
        latest_contact_force_w=point_stats["latest_contact_force_w"],
        collided_latest=point_stats["collided_latest"],
        collided_history=point_stats["collided_history"],
        contact_threshold=np.asarray([contact_threshold], dtype=np.float32),
    )


def main() -> None:
    env = None
    try:
        print(f"[INFO] Loading env config from registry for task: {args_cli.task}", flush=True)
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        env_cfg.scene.num_envs = 1

        print("[INFO] Creating gym environment...", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        print("[INFO] Resetting environment...", flush=True)
        env.reset()
        base_env = env.unwrapped

        stage = sim_utils.get_current_stage()
        collision_mesh_path = "/World/MapMesh"
        print("[INFO] Extracting drone collision envelope...", flush=True)
        envelope = _extract_drone_envelope(base_env)
        print("[INFO] Loading trajectory records...", flush=True)
        trajectory_records, trajectory_path = _load_trajectory_records(base_env)
        contact_threshold = _get_contact_threshold(env_cfg)
        output_path = Path(args_cli.output)
        points_dir = _default_points_dir(output_path)

        print("[INFO] Collision mesh:", collision_mesh_path)
        print("[INFO] Collision approximation: sdf")
        print("[INFO] Trajectory file:", trajectory_path)
        print("[INFO] Drone collision scope prim:", envelope.prim_path)
        print("[INFO] Contact threshold:", contact_threshold)
        print(
            "[INFO] Drone envelope:"
            f" xy_radius={envelope.xy_radius:.4f}m"
            f" z_rel=[{envelope.z_min_rel:.4f}, {envelope.z_max_rel:.4f}]m"
            f" sphere_radius={envelope.bounding_sphere_radius:.4f}m"
        )

        records = trajectory_records
        if args_cli.max_pairs > 0:
            records = records[: args_cli.max_pairs]

        results: list[dict[str, Any]] = []
        for pair_index, record in enumerate(records):
            if not record.get("path_reachable", False):
                continue
            if args_cli.path_source == "trajectory_xy" and not record.get("optimization_succeeded", False):
                continue

            fixed_path, path_stats = _build_fixed_height_path(
                trajectory_record=record,
                path_source=args_cli.path_source,
                path_spacing=args_cli.path_spacing,
                fixed_height=args_cli.flight_height,
                trajectory_eval_dt=args_cli.trajectory_eval_dt,
            )
            pair_stats, point_stats = _evaluate_path_with_contact_sensor(
                base_env=base_env,
                path_xyz=fixed_path,
                contact_threshold=contact_threshold,
            )
            points_file_rel = None
            if args_cli.save_point_details:
                points_file_name = (
                    f"{pair_index:04d}_"
                    f"{int(record['source_id']):02d}_{int(record['target_id']):02d}_"
                    f"{record['source_name']}_to_{record['target_name']}.npz"
                )
                points_file_name = points_file_name.replace("/", "_").replace(" ", "_")
                points_path = points_dir / points_file_name
                _save_point_details(points_path, point_stats, contact_threshold)
                points_file_rel = str(points_path)
            result = {
                "pair_index": pair_index,
                "source_id": int(record["source_id"]),
                "source_name": record["source_name"],
                "target_id": int(record["target_id"]),
                "target_name": record["target_name"],
                "collision_free": bool(pair_stats["collision_free"]),
                "path_source": args_cli.path_source,
                "flight_height": float(args_cli.flight_height),
                "num_path_points_checked": int(len(fixed_path)),
                "path_start": [float(v) for v in fixed_path[0]],
                "path_end": [float(v) for v in fixed_path[-1]],
                "original_z_span": path_stats["original_z_span"],
                "original_z_mean": path_stats["original_z_mean"],
                "first_collision": pair_stats["first_collision"],
                "max_latest_contact_force_norm": pair_stats["max_latest_contact_force_norm"],
                "max_history_contact_force_norm": pair_stats["max_history_contact_force_norm"],
                "num_collided_points_latest": pair_stats["num_collided_points_latest"],
                "num_collided_points_history": pair_stats["num_collided_points_history"],
                "points_file": points_file_rel,
            }
            results.append(result)

            status = "OK" if result["collision_free"] else "COLLISION"
            print(
                f"[{status}] {pair_index:04d}"
                f" {result['source_id']}:{result['source_name']}"
                f" -> {result['target_id']}:{result['target_name']}"
                f" points={result['num_path_points_checked']}"
                f" max_hist_force={result['max_history_contact_force_norm']:.6f}"
                f" collided_points={result['num_collided_points_history']}"
            )

        summary = _summarize_results(results)
        report = {
            "task": args_cli.task,
            "flight_height": float(args_cli.flight_height),
            "path_source": args_cli.path_source,
            "path_spacing": float(args_cli.path_spacing),
            "trajectory_eval_dt": float(args_cli.trajectory_eval_dt),
            "settle_steps": int(args_cli.settle_steps),
            "contact_threshold": float(contact_threshold),
            "collision_mesh_path": collision_mesh_path,
            "collision_approximation": "sdf",
            "trajectory_file": trajectory_path,
            "drone_envelope": {
                "prim_path": envelope.prim_path,
                "root_path": envelope.root_path,
                "xy_radius": envelope.xy_radius,
                "z_min_rel": envelope.z_min_rel,
                "z_max_rel": envelope.z_max_rel,
                "bounding_sphere_radius": envelope.bounding_sphere_radius,
                "bbox_min_world": list(envelope.bbox_min_world),
                "bbox_max_world": list(envelope.bbox_max_world),
            },
            "points_dir": str(points_dir) if args_cli.save_point_details else None,
            "summary": summary,
            "results": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print("[SUMMARY]", json.dumps(summary, indent=2, ensure_ascii=False))
        print("[INFO] Report saved to", output_path)
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

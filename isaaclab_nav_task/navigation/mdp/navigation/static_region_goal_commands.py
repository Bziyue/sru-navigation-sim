"""Goal command generator for region-based navigation on a static scan mesh."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.markers.config import CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms, transform_points, yaw_quat
from pxr import UsdGeom
from scipy.spatial import KDTree

from isaaclab_nav_task.navigation.mdp.math_utils import vec_to_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .static_region_goal_commands_cfg import StaticRegionGoalCommandCfg


_RECT_NAME_RE = re.compile(r"^Rectangle:\s*(.+)$")
_CORNER_RE = re.compile(r"Corner\s+\d+:\s+X:\s*([-+0-9.]+),\s*Y:\s*([-+0-9.]+),\s*Z:\s*([-+0-9.]+)")
_REGION_POINT_SET_CACHE: dict[tuple, tuple[list[dict], np.ndarray]] = {}


class SuccessRateTracker:
    """Tracks command success in a rolling per-environment buffer."""

    def __init__(self, num_envs: int, device: torch.device, buffer_size: int = 10):
        self.buffer = torch.full((num_envs, buffer_size), -1.0, device=device)
        self.write_index = torch.zeros(num_envs, dtype=torch.long, device=device)

    def add(self, results: torch.Tensor, env_ids: torch.Tensor):
        indices = self.write_index[env_ids] % self.buffer.shape[1]
        self.buffer[env_ids, indices] = results[env_ids].float()
        self.write_index[env_ids] += 1

    def get_success_rate(self) -> torch.Tensor:
        filled_count = (self.buffer >= 0).sum(dim=1).clamp(min=1)
        success_count = (self.buffer > 0).sum(dim=1)
        return success_count.float() / filled_count.float()


def _load_region_boxes(path: str) -> list[dict]:
    regions: list[dict] = []
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
                    points = torch.tensor(current_points, dtype=torch.float32)
                    xy_min = points[:, :2].min(dim=0).values.numpy().astype(np.float32, copy=False)
                    xy_max = points[:, :2].max(dim=0).values.numpy().astype(np.float32, copy=False)
                    regions.append(
                        {
                            "name": current_name,
                            "xy_min": xy_min,
                            "xy_max": xy_max,
                            "center_xy": ((xy_min + xy_max) * 0.5).astype(np.float32, copy=False),
                            "floor_z": float(points[:, 2].min().item()),
                        }
                    )
                current_name = match.group(1)
                current_points = []
                continue

            match = _CORNER_RE.search(line)
            if match is not None:
                current_points.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])

    if current_name is not None and current_points:
        points = torch.tensor(current_points, dtype=torch.float32)
        xy_min = points[:, :2].min(dim=0).values.numpy().astype(np.float32, copy=False)
        xy_max = points[:, :2].max(dim=0).values.numpy().astype(np.float32, copy=False)
        regions.append(
            {
                "name": current_name,
                "xy_min": xy_min,
                "xy_max": xy_max,
                "center_xy": ((xy_min + xy_max) * 0.5).astype(np.float32, copy=False),
                "floor_z": float(points[:, 2].min().item()),
            }
        )

    if not regions:
        raise ValueError(f"No region rectangles found in {path}")

    return regions


def _triangulate_faces(face_vertex_counts, face_vertex_indices) -> np.ndarray:
    faces = []
    index = 0
    for count in face_vertex_counts:
        for offset in range(int(count) - 2):
            faces.append(
                [
                    int(face_vertex_indices[index]),
                    int(face_vertex_indices[index + 1 + offset]),
                    int(face_vertex_indices[index + 2 + offset]),
                ]
            )
        index += int(count)
    return np.asarray(faces, dtype=np.int32)


def _trimesh_from_stage_mesh(mesh_prim_path: str) -> trimesh.Trimesh:
    stage = get_current_stage()
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise ValueError(f"Map mesh prim not found: {mesh_prim_path}")

    mesh_geom = UsdGeom.Mesh(mesh_prim)
    vertices = mesh_geom.GetPointsAttr().Get()
    face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
    if not vertices or not face_vertex_counts or not face_vertex_indices:
        raise ValueError(f"Map mesh prim is empty: {mesh_prim_path}")

    vertices_np = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
    faces_np = _triangulate_faces(face_vertex_counts, face_vertex_indices)
    return trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)


def _build_occupied_kdtree(mesh: trimesh.Trimesh, num_surface_samples: int = 150000) -> KDTree:
    try:
        surface_points, _ = trimesh.sample.sample_surface_even(mesh, num_surface_samples)
    except Exception:
        surface_points = np.asarray(mesh.vertices, dtype=np.float32)
    return KDTree(np.asarray(surface_points, dtype=np.float32))


def _build_uniform_axis(low: float, high: float, spacing: float) -> np.ndarray:
    span = float(high - low)
    if span <= spacing:
        return np.asarray([(low + high) * 0.5], dtype=np.float32)

    axis = np.arange(low + 0.5 * spacing, high, spacing, dtype=np.float32)
    if len(axis) == 0:
        axis = np.asarray([(low + high) * 0.5], dtype=np.float32)
    return axis


def _sample_region_safe_grid_points(
    region: dict,
    occupied_kdtree: KDTree,
    flight_height: float,
    point_clearance: float,
    grid_spacing: float,
) -> np.ndarray:
    xy_min = np.asarray(region["xy_min"], dtype=np.float32)
    xy_max = np.asarray(region["xy_max"], dtype=np.float32)
    x_axis = _build_uniform_axis(float(xy_min[0]), float(xy_max[0]), grid_spacing)
    y_axis = _build_uniform_axis(float(xy_min[1]), float(xy_max[1]), grid_spacing)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="xy")
    candidate_points = np.column_stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.full(grid_x.size, flight_height, dtype=np.float32),
        )
    ).astype(np.float32, copy=False)

    distances, _ = occupied_kdtree.query(candidate_points, k=1)
    return candidate_points[np.asarray(distances) >= point_clearance].astype(np.float32, copy=False)


def _load_precomputed_region_point_sets(
    surface_bbox_data_path: str,
    precomputed_safe_points_path: str,
    flight_height: float,
) -> tuple[list[dict], np.ndarray]:
    base_regions = _load_region_boxes(surface_bbox_data_path)
    payload = np.load(precomputed_safe_points_path)

    region_names = np.asarray(payload["region_names"]).astype(str).tolist()
    region_xy_min = np.asarray(payload["region_xy_min"], dtype=np.float32)
    region_xy_max = np.asarray(payload["region_xy_max"], dtype=np.float32)
    region_floor_z = np.asarray(payload["region_floor_z"], dtype=np.float32)
    region_start_indices = np.asarray(payload["region_start_indices"], dtype=np.int64)
    region_counts = np.asarray(payload["region_counts"], dtype=np.int64)
    points_xyz = np.asarray(payload["points_xyz"], dtype=np.float32)

    if len(base_regions) != len(region_names):
        raise RuntimeError(
            "Precomputed safe point file does not match the configured region box count: "
            f"{len(region_names)} vs {len(base_regions)}."
        )

    loaded_regions: list[dict] = []
    all_safe_points: list[np.ndarray] = []
    for region_id, base_region in enumerate(base_regions):
        if base_region["name"] != region_names[region_id]:
            raise RuntimeError(
                "Region order mismatch between bbox file and precomputed safe point file: "
                f"bbox[{region_id}]={base_region['name']} vs safe_points[{region_id}]={region_names[region_id]}."
            )

        if not np.allclose(base_region["xy_min"], region_xy_min[region_id], atol=1e-3):
            raise RuntimeError(f"Region xy_min mismatch for region {region_id}:{base_region['name']}.")
        if not np.allclose(base_region["xy_max"], region_xy_max[region_id], atol=1e-3):
            raise RuntimeError(f"Region xy_max mismatch for region {region_id}:{base_region['name']}.")
        if not np.isclose(base_region["floor_z"], region_floor_z[region_id], atol=1e-3):
            raise RuntimeError(f"Region floor_z mismatch for region {region_id}:{base_region['name']}.")

        start = int(region_start_indices[region_id])
        count = int(region_counts[region_id])
        safe_points = points_xyz[start : start + count].astype(np.float32, copy=False)
        region = dict(base_region)
        center_z = float(np.mean(safe_points[:, 2])) if len(safe_points) > 0 else float(flight_height)
        region["center"] = np.array([region["center_xy"][0], region["center_xy"][1], center_z], dtype=np.float32)
        region["safe_points"] = safe_points
        loaded_regions.append(region)
        if len(safe_points) > 0:
            all_safe_points.append(safe_points)

    if not all_safe_points:
        raise RuntimeError(f"No safe points were loaded from {precomputed_safe_points_path}.")

    safe_points = np.concatenate(all_safe_points, axis=0)
    return loaded_regions, safe_points.astype(np.float32, copy=False)


def _build_region_point_sets(
    surface_bbox_data_path: str,
    map_mesh_prim_path: str,
    flight_height: float,
    point_clearance: float,
    grid_spacing: float,
    precomputed_safe_points_path: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    cache_key = (
        surface_bbox_data_path,
        map_mesh_prim_path,
        float(flight_height),
        float(point_clearance),
        float(grid_spacing),
        precomputed_safe_points_path,
    )
    if cache_key in _REGION_POINT_SET_CACHE:
        return _REGION_POINT_SET_CACHE[cache_key]

    if precomputed_safe_points_path is not None:
        region_point_sets, safe_points = _load_precomputed_region_point_sets(
            surface_bbox_data_path=surface_bbox_data_path,
            precomputed_safe_points_path=precomputed_safe_points_path,
            flight_height=flight_height,
        )
        _REGION_POINT_SET_CACHE[cache_key] = (region_point_sets, safe_points)
        return region_point_sets, safe_points

    mesh = _trimesh_from_stage_mesh(map_mesh_prim_path)
    occupied_kdtree = _build_occupied_kdtree(mesh)
    regions = _load_region_boxes(surface_bbox_data_path)

    all_safe_points: list[np.ndarray] = []
    for region in regions:
        points = _sample_region_safe_grid_points(
            region=region,
            occupied_kdtree=occupied_kdtree,
            flight_height=flight_height,
            point_clearance=point_clearance,
            grid_spacing=grid_spacing,
        )
        region["center"] = np.array([region["center_xy"][0], region["center_xy"][1], flight_height], dtype=np.float32)
        region["safe_points"] = points
        if len(points) > 0:
            all_safe_points.append(points)

    if not all_safe_points:
        raise RuntimeError(f"No safe points were generated from {surface_bbox_data_path}.")

    safe_points = np.concatenate(all_safe_points, axis=0)
    _, unique_indices = np.unique(np.round(safe_points, decimals=4), axis=0, return_index=True)
    safe_points = safe_points[np.sort(unique_indices)].astype(np.float32, copy=False)

    if len(safe_points) < 2:
        raise RuntimeError(
            "Expected at least two safe points from "
            f"{surface_bbox_data_path}, but got {len(safe_points)}."
        )

    _REGION_POINT_SET_CACHE[cache_key] = (regions, safe_points)
    return regions, safe_points


def _sample_quintic_centerline(
    breakpoints: list[float],
    coefficients: list[list[float]],
    eval_dt: float,
    arc_length_spacing: float,
    flight_height: float,
) -> np.ndarray:
    num_segments = len(breakpoints) - 1
    if num_segments <= 0:
        raise ValueError("Trajectory must contain at least one segment.")
    if len(coefficients) != num_segments * 6:
        raise ValueError(
            f"Expected {num_segments * 6} coefficient rows for {num_segments} segments, got {len(coefficients)}."
        )

    sampled_points: list[np.ndarray] = []
    coeffs_np = np.asarray(coefficients, dtype=np.float32).reshape(num_segments, 6, 3)

    for seg_idx in range(num_segments):
        duration = float(breakpoints[seg_idx + 1] - breakpoints[seg_idx])
        if duration <= 0.0:
            continue

        local_times = np.arange(0.0, duration, eval_dt, dtype=np.float32)
        if seg_idx == num_segments - 1 or len(local_times) == 0 or not np.isclose(local_times[-1], duration):
            local_times = np.concatenate([local_times, np.asarray([duration], dtype=np.float32)])

        powers = np.stack([np.power(local_times, power, dtype=np.float32) for power in range(6)], axis=1)
        segment_points = np.einsum("tk,kc->tc", powers, coeffs_np[seg_idx]).astype(np.float32, copy=False)
        sampled_points.append(segment_points)

    if not sampled_points:
        raise ValueError("Failed to sample any points from trajectory coefficients.")

    dense_centerline = np.concatenate(sampled_points, axis=0)
    dense_centerline[:, 2] = flight_height
    _, unique_indices = np.unique(np.round(dense_centerline[:, :2], decimals=4), axis=0, return_index=True)
    dense_centerline = dense_centerline[np.sort(unique_indices)]
    if len(dense_centerline) < 2:
        raise ValueError("Sampled trajectory centerline must contain at least two distinct points.")

    segment_vectors = np.diff(dense_centerline[:, :2], axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_lengths = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)],
        axis=0,
    )
    total_length = float(cumulative_lengths[-1])
    if total_length <= 1e-6:
        raise ValueError("Trajectory centerline length is too small after dense sampling.")

    sample_arcs = np.arange(0.0, total_length, arc_length_spacing, dtype=np.float32)
    if len(sample_arcs) == 0 or not np.isclose(sample_arcs[-1], total_length):
        sample_arcs = np.concatenate([sample_arcs, np.asarray([total_length], dtype=np.float32)])

    resampled_xy = np.column_stack(
        [
            np.interp(sample_arcs, cumulative_lengths, dense_centerline[:, axis]).astype(np.float32, copy=False)
            for axis in range(2)
        ]
    ).astype(np.float32, copy=False)
    centerline = np.zeros((len(resampled_xy), 3), dtype=np.float32)
    centerline[:, :2] = resampled_xy
    centerline[:, 2] = flight_height
    return centerline


def _load_guidance_centerlines(
    guidance_trajectories_data_path: str,
    flight_height: float,
    eval_dt: float,
    arc_length_spacing: float,
    num_regions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(guidance_trajectories_data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    trajectories = payload.get("trajectories", [])
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(f"No trajectories found in {guidance_trajectories_data_path}")

    directed_paths: list[np.ndarray] = []
    directed_arcs: list[np.ndarray] = []
    directed_pairs: list[tuple[int, int]] = []

    for trajectory in trajectories:
        if not trajectory.get("path_reachable", False):
            continue
        if not trajectory.get("optimization_succeeded", False):
            continue

        source_id = int(trajectory["source_id"])
        target_id = int(trajectory["target_id"])
        centerline = _sample_quintic_centerline(
            breakpoints=list(trajectory["breakpoints"]),
            coefficients=list(trajectory["coefficients"]),
            eval_dt=eval_dt,
            arc_length_spacing=arc_length_spacing,
            flight_height=flight_height,
        )
        segment_lengths = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
        arc_lengths = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)], axis=0)

        directed_pairs.append((source_id, target_id))
        directed_paths.append(centerline[:, :2].astype(np.float32, copy=False))
        directed_arcs.append(arc_lengths.astype(np.float32, copy=False))

        directed_pairs.append((target_id, source_id))
        directed_paths.append(centerline[::-1, :2].astype(np.float32, copy=False))
        reversed_arcs = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths[::-1], dtype=np.float32)],
            axis=0,
        )
        directed_arcs.append(reversed_arcs.astype(np.float32, copy=False))

    if not directed_paths:
        raise RuntimeError(f"No usable optimized trajectories found in {guidance_trajectories_data_path}")

    max_len = max(len(path) for path in directed_paths)
    padded_paths = np.zeros((len(directed_paths), max_len, 2), dtype=np.float32)
    padded_arcs = np.zeros((len(directed_paths), max_len), dtype=np.float32)
    path_lengths = np.zeros(len(directed_paths), dtype=np.int64)
    pair_lookup = np.full((num_regions, num_regions), -1, dtype=np.int64)

    for idx, ((source_id, target_id), path_xy, arc_lengths) in enumerate(zip(directed_pairs, directed_paths, directed_arcs, strict=True)):
        length = len(path_xy)
        padded_paths[idx, :length] = path_xy
        padded_arcs[idx, :length] = arc_lengths
        path_lengths[idx] = length
        pair_lookup[source_id, target_id] = idx

    if np.any(pair_lookup[np.triu_indices(num_regions, k=1)] < 0):
        raise RuntimeError("Missing directed guidance paths for some region pairs.")

    pair_ids = np.asarray(directed_pairs, dtype=np.int64)
    return (
        torch.as_tensor(padded_paths, dtype=torch.float32),
        torch.as_tensor(padded_arcs, dtype=torch.float32),
        torch.as_tensor(path_lengths, dtype=torch.long),
        torch.as_tensor(pair_ids, dtype=torch.long),
    )


class StaticRegionGoalCommand(CommandTerm):
    """Sample spawn and goal points from predefined mesh regions."""

    cfg: StaticRegionGoalCommandCfg

    def __init__(self, cfg: StaticRegionGoalCommandCfg, env: ManagerBasedRLEnv):
        self.visualize_region_safe_points = bool(cfg.visualize_region_safe_points)
        self.region_safe_points_vis: torch.Tensor | None = None
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        region_point_sets, safe_points = _build_region_point_sets(
            surface_bbox_data_path=cfg.surface_bbox_data_path,
            map_mesh_prim_path=cfg.map_mesh_prim_path,
            flight_height=float(cfg.flight_height),
            point_clearance=float(cfg.point_clearance),
            grid_spacing=float(cfg.safe_point_grid_spacing),
            precomputed_safe_points_path=cfg.precomputed_safe_points_path,
        )
        self.region_names = [region["name"] for region in region_point_sets]
        self.region_xy_min = torch.as_tensor(
            np.asarray([region["xy_min"] for region in region_point_sets], dtype=np.float32), device=self.device
        )
        self.region_xy_max = torch.as_tensor(
            np.asarray([region["xy_max"] for region in region_point_sets], dtype=np.float32), device=self.device
        )
        region_centers = np.asarray([region["center"] for region in region_point_sets], dtype=np.float32)
        self.region_centers = torch.as_tensor(region_centers[:, :2], device=self.device)
        self.safe_point_pool = torch.as_tensor(safe_points, dtype=torch.float32, device=self.device)
        self.region_safe_points = [
            torch.as_tensor(region["safe_points"], dtype=torch.float32, device=self.device) for region in region_point_sets
        ]
        self.flight_height = float(cfg.flight_height)
        self.visualize_region_boxes = bool(cfg.visualize_region_boxes)
        self.guidance_eval_dt = float(cfg.guidance_trajectory_eval_dt)
        self.guidance_arc_length_spacing = float(cfg.guidance_arc_length_spacing)

        if cfg.guidance_trajectories_data_path is None:
            raise ValueError("guidance_trajectories_data_path must be provided for static region guidance.")
        (
            guidance_paths_xy,
            guidance_arc_lengths,
            guidance_path_lengths,
            guidance_pair_ids,
        ) = _load_guidance_centerlines(
            guidance_trajectories_data_path=cfg.guidance_trajectories_data_path,
            flight_height=self.flight_height,
            eval_dt=self.guidance_eval_dt,
            arc_length_spacing=self.guidance_arc_length_spacing,
            num_regions=len(region_point_sets),
        )
        self.guidance_paths_xy = guidance_paths_xy.to(self.device)
        self.guidance_arc_lengths = guidance_arc_lengths.to(self.device)
        self.guidance_path_lengths = guidance_path_lengths.to(self.device)
        self.guidance_pair_ids = guidance_pair_ids.to(self.device)
        self.num_precomputed_guidance_paths = int(self.guidance_pair_ids.shape[0])
        self.num_precomputed_region_pairs = self.num_precomputed_guidance_paths // 2
        self._validate_guidance_region_alignment()
        print(
            "[StaticRegionGoalCommand] Precomputed "
            f"{self.num_precomputed_guidance_paths} directed guidance trajectories "
            f"({self.num_precomputed_region_pairs} undirected region pairs). "
            "These centerlines are loaded once at initialization and reused by indexing during training."
        )
        self.region_safe_points_vis, self.region_safe_points_vis_indices = self._build_region_safe_points_vis(
            all_regions=region_point_sets,
            points_per_region=int(cfg.region_safe_points_vis_points_per_region),
        )
        self.region_box_vis_translations, self.region_box_vis_scales = self._build_region_box_vis(
            all_regions=region_point_sets,
            box_height=float(cfg.region_box_vis_height),
            z_offset=float(cfg.region_vis_z_offset),
        )
        if self.cfg.debug_vis:
            self._set_debug_vis_impl(True)

        self.goal_command_body = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_command_body_unscaled = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_position_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_position_world[:, 2] = self.flight_height
        self.spawn_position_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.spawn_position_world[:, 2] = self.flight_height
        self.spawn_heading_world = torch.zeros(self.num_envs, device=self.device)
        self.spawn_region_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_region_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_guidance_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.guidance_progress = torch.zeros(self.num_envs, device=self.device)
        self.previous_guidance_progress = torch.zeros(self.num_envs, device=self.device)
        self.guidance_progress_delta = torch.zeros(self.num_envs, device=self.device)
        self.guidance_lateral_error = torch.zeros(self.num_envs, device=self.device)
        self.guidance_vis_point_cap = 160

        self.steps_at_goal = torch.zeros(self.num_envs, device=self.device)
        self.time_at_goal = torch.zeros(self.num_envs, device=self.device)
        self.required_steps_at_goal = 4.0 / self.env.step_dt
        self.initial_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self.closest_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self.total_distance_traveled = torch.zeros(self.num_envs, device=self.device)
        self.previous_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.success_tracker = SuccessRateTracker(self.num_envs, self.device, buffer_size=10)
        self.success_rate_buffer = torch.full((self.num_envs, 10), -1.0, device=self.device)

        self.metrics["velocity_toward_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["velocity_magnitude"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["active_guidance_assignments"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["active_guidance_unique"] = torch.zeros(self.num_envs, device=self.device)

    def _build_region_safe_points_vis(
        self,
        all_regions: list[dict],
        points_per_region: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.visualize_region_safe_points:
            return None, None

        vis_points: list[torch.Tensor] = []
        vis_indices: list[torch.Tensor] = []
        for region in all_regions:
            if len(region["safe_points"]) > 0:
                points = torch.as_tensor(region["safe_points"], dtype=torch.float32, device=self.device)
                if points_per_region > 0 and len(points) > points_per_region:
                    indices = torch.linspace(0, len(points) - 1, points_per_region, device=self.device)
                    points = points[indices.round().long()]
                points = points.clone()
                points[:, 2] += float(self.cfg.region_vis_z_offset)
                vis_points.append(points)
                vis_indices.append(torch.zeros(len(points), dtype=torch.long, device=self.device))
                continue

            placeholder_height = max(float(region["floor_z"]) + float(self.cfg.region_vis_z_offset), self.flight_height + 0.2)
            placeholder = torch.tensor(
                [[float(region["center_xy"][0]), float(region["center_xy"][1]), placeholder_height]],
                dtype=torch.float32,
                device=self.device,
            )
            vis_points.append(placeholder)
            vis_indices.append(torch.ones(1, dtype=torch.long, device=self.device))

        if not vis_points:
            return None, None
        return torch.cat(vis_points, dim=0), torch.cat(vis_indices, dim=0)

    def _build_region_box_vis(
        self,
        all_regions: list[dict],
        box_height: float,
        z_offset: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.visualize_region_boxes:
            return None, None

        xy_min = torch.as_tensor(np.asarray([region["xy_min"] for region in all_regions], dtype=np.float32), device=self.device)
        xy_max = torch.as_tensor(np.asarray([region["xy_max"] for region in all_regions], dtype=np.float32), device=self.device)
        floor_z = torch.as_tensor(np.asarray([region["floor_z"] for region in all_regions], dtype=np.float32), device=self.device)
        centers_xy = 0.5 * (xy_min + xy_max)
        translations = torch.zeros((len(all_regions), 3), dtype=torch.float32, device=self.device)
        translations[:, :2] = centers_xy
        translations[:, 2] = floor_z + z_offset

        scales = torch.ones((len(all_regions), 3), dtype=torch.float32, device=self.device)
        scales[:, :2] = torch.clamp(xy_max - xy_min, min=0.05)
        scales[:, 2] = max(box_height, 0.01)
        return translations, scales

    @property
    def command(self) -> torch.Tensor:
        return self.goal_command_body

    def _get_unscaled_command(self) -> torch.Tensor:
        return self.goal_command_body_unscaled

    def _sample_safe_points_from_regions(self, region_ids: torch.Tensor) -> torch.Tensor:
        sampled_points = torch.zeros((len(region_ids), 3), dtype=torch.float32, device=self.device)
        for local_idx, region_id in enumerate(region_ids.tolist()):
            region_points = self.region_safe_points[region_id]
            if len(region_points) == 0:
                raise RuntimeError(f"Region {region_id} does not contain any safe spawn points.")
            point_index = torch.randint(0, len(region_points), (1,), device=self.device).item()
            sampled_points[local_idx] = region_points[point_index]
        return sampled_points

    def _points_inside_regions(
        self,
        points_xy: torch.Tensor,
        region_ids: torch.Tensor,
        tolerance: float = 1e-4,
    ) -> torch.Tensor:
        """Check whether points lie inside their corresponding region boxes."""
        xy_min = self.region_xy_min[region_ids] - tolerance
        xy_max = self.region_xy_max[region_ids] + tolerance
        return torch.all((points_xy >= xy_min) & (points_xy <= xy_max), dim=1)

    def _validate_guidance_region_alignment(self) -> None:
        """Validate that guidance endpoints and safe point pools match the region pairing."""
        referenced_region_ids = torch.unique(self.guidance_pair_ids.reshape(-1))
        empty_region_ids = [
            int(region_id.item()) for region_id in referenced_region_ids if len(self.region_safe_points[int(region_id.item())]) == 0
        ]
        if empty_region_ids:
            empty_region_names = [self.region_names[region_id] for region_id in empty_region_ids]
            raise RuntimeError(
                "Guidance trajectories reference regions without any safe spawn/goal points: "
                + ", ".join(
                    f"{region_id}:{region_name}"
                    for region_id, region_name in zip(empty_region_ids, empty_region_names, strict=True)
                )
                )

        same_region_ids = torch.where(self.guidance_pair_ids[:, 0] == self.guidance_pair_ids[:, 1])[0]
        if len(same_region_ids) > 0:
            example_id = int(same_region_ids[0].item())
            region_id = int(self.guidance_pair_ids[example_id, 0].item())
            raise RuntimeError(
                "Guidance trajectory pairs must connect two different regions. "
                f"Example invalid path id {example_id}: {region_id}:{self.region_names[region_id]}."
            )

        path_ids = torch.arange(len(self.guidance_pair_ids), device=self.device)
        start_points = self.guidance_paths_xy[path_ids, 0]
        end_indices = self.guidance_path_lengths - 1
        end_points = self.guidance_paths_xy[path_ids, end_indices]
        source_region_ids = self.guidance_pair_ids[:, 0]
        target_region_ids = self.guidance_pair_ids[:, 1]

        start_ok = self._points_inside_regions(start_points, source_region_ids)
        end_ok = self._points_inside_regions(end_points, target_region_ids)
        invalid_ids = torch.where(~(start_ok & end_ok))[0]
        if len(invalid_ids) > 0:
            example_id = int(invalid_ids[0].item())
            source_id = int(source_region_ids[example_id].item())
            target_id = int(target_region_ids[example_id].item())
            raise RuntimeError(
                "Guidance trajectory endpoints do not match their source/target regions. "
                f"Example path id {example_id}: "
                f"{source_id}:{self.region_names[source_id]} -> {target_id}:{self.region_names[target_id]}."
            )

    def _sample_region_pairs(self, num_pairs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample directed guidance pairs, ensuring spawn and goal belong to different regions."""
        sampled_guidance_indices = torch.randint(0, len(self.guidance_pair_ids), (num_pairs,), device=self.device)
        sampled_pairs = self.guidance_pair_ids[sampled_guidance_indices]
        source_region_ids = sampled_pairs[:, 0]
        target_region_ids = sampled_pairs[:, 1]
        if torch.any(source_region_ids == target_region_ids):
            same_region_ids = torch.where(source_region_ids == target_region_ids)[0]
            example_local_idx = int(same_region_ids[0].item())
            region_id = int(source_region_ids[example_local_idx].item())
            raise RuntimeError(
                "Sampled a guidance pair with identical source and target regions, which is not allowed. "
                f"Example local index {example_local_idx}: {region_id}:{self.region_names[region_id]}."
            )
        return sampled_guidance_indices, source_region_ids, target_region_ids

    def _apply_spawn_state_to_robot(self, env_ids: torch.Tensor, spawn_points: torch.Tensor) -> None:
        """Write the freshly sampled spawn positions back to the robot root state.

        The environment reset events run before command resampling, so updating only
        ``env.scene.env_origins`` is too late for the robot pose reset in the same
        episode. We therefore keep the sampled yaw from the earlier reset event but
        overwrite the root position and zero the root velocity here so the physical
        robot starts from the same spawn point used by the command term.
        """
        current_root_quat = self.robot.data.root_quat_w[env_ids].clone()
        root_pose = torch.cat((spawn_points, current_root_quat), dim=-1)
        root_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        self.spawn_heading_world[env_ids] = math_utils.euler_xyz_from_quat(current_root_quat)[2]
        spawn_reset_error = torch.norm(self.robot.data.root_pos_w[env_ids] - spawn_points, dim=1)
        if torch.any(spawn_reset_error > 1e-4):
            max_error = float(torch.max(spawn_reset_error).item())
            raise RuntimeError(
                "Failed to align robot root pose with sampled spawn positions during reset. "
                f"Maximum position error: {max_error:.6f} m."
            )

    def _project_guidance_state(self, positions_xy: torch.Tensor, guidance_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project positions onto the current guidance polyline using segment-wise projection.

        This is more stable than snapping to the nearest sampled waypoint, especially around
        bends and densely sampled regions, because progress is measured continuously along the
        polyline rather than jumping between discrete vertices.
        """
        path_xy = self.guidance_paths_xy[guidance_ids]
        arc_lengths = self.guidance_arc_lengths[guidance_ids]
        path_lengths = self.guidance_path_lengths[guidance_ids]

        start_points = path_xy[:, :-1, :]
        end_points = path_xy[:, 1:, :]
        segment_vectors = end_points - start_points
        segment_lengths_sq = torch.sum(segment_vectors * segment_vectors, dim=-1).clamp_min(1e-9)
        segment_lengths = torch.sqrt(segment_lengths_sq)

        rel_positions = positions_xy.unsqueeze(1) - start_points
        raw_t = torch.sum(rel_positions * segment_vectors, dim=-1) / segment_lengths_sq
        clamped_t = torch.clamp(raw_t, min=0.0, max=1.0)
        projected_points = start_points + clamped_t.unsqueeze(-1) * segment_vectors

        deltas = positions_xy.unsqueeze(1) - projected_points
        distances_sq = torch.sum(deltas * deltas, dim=-1)

        max_segments = start_points.shape[1]
        valid_segment_mask = torch.arange(max_segments, device=self.device).unsqueeze(0) < (path_lengths - 1).unsqueeze(1)
        distances_sq = torch.where(valid_segment_mask, distances_sq, torch.full_like(distances_sq, float("inf")))

        closest_segment_indices = torch.argmin(distances_sq, dim=1)
        lateral_error = torch.sqrt(
            torch.gather(distances_sq, 1, closest_segment_indices.unsqueeze(1)).squeeze(1).clamp_min(0.0)
        )
        segment_start_arc = torch.gather(arc_lengths[:, :-1], 1, closest_segment_indices.unsqueeze(1)).squeeze(1)
        closest_segment_t = torch.gather(clamped_t, 1, closest_segment_indices.unsqueeze(1)).squeeze(1)
        closest_segment_length = torch.gather(segment_lengths, 1, closest_segment_indices.unsqueeze(1)).squeeze(1)
        progress = segment_start_arc + closest_segment_t * closest_segment_length
        return progress, lateral_error

    def _get_guidance_vis_points_for_env(self, env_index: int = 0) -> torch.Tensor | None:
        if self.num_envs <= 0:
            return None

        guidance_id = int(self.current_guidance_ids[env_index].item())
        path_length = int(self.guidance_path_lengths[guidance_id].item())
        if path_length <= 0:
            return None

        path_xy = self.guidance_paths_xy[guidance_id, :path_length]
        if path_length > self.guidance_vis_point_cap:
            sample_indices = torch.linspace(0, path_length - 1, self.guidance_vis_point_cap, device=self.device)
            path_xy = path_xy[sample_indices.round().long()]

        vis_points = torch.zeros((len(path_xy), 3), dtype=torch.float32, device=self.device)
        vis_points[:, :2] = path_xy
        vis_points[:, 2] = self.flight_height + 0.12
        return vis_points

    def _update_metrics(self):
        position_error = self.goal_position_world - self.robot.data.root_pos_w[:, :3]
        position_error_2d = position_error[:, :2]
        velocity_2d = self.robot.data.root_state_w[:, 7:9]

        self.metrics["velocity_magnitude"] = torch.norm(velocity_2d, dim=1)
        direction_to_goal = position_error_2d / torch.clamp(torch.norm(position_error_2d, dim=1, keepdim=True), min=1e-6)
        self.metrics["velocity_toward_goal"] = (velocity_2d * direction_to_goal).sum(dim=1)
        self.metrics["success_rate"] = self.success_tracker.get_success_rate()
        self.metrics["goal_z"] = self.goal_position_world[:, 2]
        active_assignments = float((self.current_guidance_ids >= 0).sum().item())
        active_unique = float(torch.unique(self.current_guidance_ids).numel())
        self.metrics["active_guidance_assignments"].fill_(active_assignments)
        self.metrics["active_guidance_unique"].fill_(active_unique)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.steps_at_goal[env_ids_tensor] = 0
        self.time_at_goal[env_ids_tensor] = 0
        self.total_distance_traveled[env_ids_tensor] = 0.0

        sampled_guidance_indices, source_region_ids, target_region_ids = self._sample_region_pairs(len(env_ids_tensor))

        spawn_points = self._sample_safe_points_from_regions(source_region_ids)
        goal_points = self._sample_safe_points_from_regions(target_region_ids)
        if not torch.all(self._points_inside_regions(spawn_points[:, :2], source_region_ids)):
            raise RuntimeError("Sampled spawn points do not lie inside their source regions.")
        if not torch.all(self._points_inside_regions(goal_points[:, :2], target_region_ids)):
            raise RuntimeError("Sampled goal points do not lie inside their target regions.")
        self.spawn_region_ids[env_ids_tensor] = source_region_ids
        self.goal_region_ids[env_ids_tensor] = target_region_ids
        self.current_guidance_ids[env_ids_tensor] = sampled_guidance_indices
        self.spawn_position_world[env_ids_tensor] = spawn_points
        self.goal_position_world[env_ids_tensor] = goal_points
        self.env.scene.env_origins[env_ids_tensor] = spawn_points
        self._apply_spawn_state_to_robot(env_ids_tensor, spawn_points)
        self.previous_position[env_ids_tensor] = spawn_points
        self.initial_distance_to_goal[env_ids_tensor] = torch.norm(goal_points - spawn_points, dim=1)
        self.closest_distance_to_goal[env_ids_tensor] = self.initial_distance_to_goal[env_ids_tensor]
        start_progress, start_lateral_error = self._project_guidance_state(
            positions_xy=spawn_points[:, :2],
            guidance_ids=sampled_guidance_indices,
        )
        self.guidance_progress[env_ids_tensor] = start_progress
        self.previous_guidance_progress[env_ids_tensor] = start_progress
        self.guidance_progress_delta[env_ids_tensor] = 0.0
        self.guidance_lateral_error[env_ids_tensor] = start_lateral_error

    def _update_command(self):
        inverse_pos, inverse_rot = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w)
        goal_in_body = transform_points(self.goal_position_world.unsqueeze(1), inverse_pos, inverse_rot).squeeze(1)

        self.goal_command_body_unscaled = goal_in_body

        distance = torch.norm(goal_in_body, dim=-1, keepdim=True)
        direction = goal_in_body / torch.clamp(distance, min=1e-6)
        log_distance = torch.log(distance + 1.0)
        self.goal_command_body[:, :3] = direction
        self.goal_command_body[:, 3:] = log_distance

        self.distance_to_goal = torch.norm(self.robot.data.root_pos_w - self.goal_position_world, dim=1)
        self.closest_distance_to_goal = torch.min(self.closest_distance_to_goal, self.distance_to_goal)
        step_distance = torch.norm(self.robot.data.root_pos_w - self.previous_position, dim=1)
        self.total_distance_traveled += step_distance
        self.previous_position = self.robot.data.root_pos_w.clone()
        guidance_progress, guidance_lateral_error = self._project_guidance_state(
            positions_xy=self.robot.data.root_pos_w[:, :2],
            guidance_ids=self.current_guidance_ids,
        )
        self.previous_guidance_progress.copy_(self.guidance_progress)
        self.guidance_progress.copy_(guidance_progress)
        self.guidance_progress_delta.copy_(self.guidance_progress - self.previous_guidance_progress)
        self.guidance_lateral_error.copy_(guidance_lateral_error)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        self.command_counter[env_ids] = 0
        self._resample(env_ids)

        extras = {}
        for name, value in self.metrics.items():
            extras[name] = torch.mean(value[env_ids]).item()
            value[env_ids] = 0.0
        return extras

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_marker"):
                cfg = CUBOID_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/goal_position"
                cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                self.goal_marker = VisualizationMarkers(cfg)

            if not hasattr(self, "spawn_marker"):
                cfg = CUBOID_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/spawn_position"
                cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.5, 0.0)
                self.spawn_marker = VisualizationMarkers(cfg)

            if not hasattr(self, "desired_velocity_marker"):
                cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/desired_velocity"
                cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.desired_velocity_marker = VisualizationMarkers(cfg)

            if not hasattr(self, "current_velocity_marker"):
                cfg = RED_ARROW_X_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/current_velocity"
                cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.current_velocity_marker = VisualizationMarkers(cfg)

            if getattr(self, "region_safe_points_vis", None) is not None and not hasattr(self, "region_safe_points_marker"):
                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/region_safe_points",
                    markers={
                        "point": sim_utils.SphereCfg(
                            radius=0.04,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.85, 0.85),
                                opacity=0.35,
                            ),
                        )
                        ,
                        "empty_region": sim_utils.SphereCfg(
                            radius=0.08,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.2, 0.2),
                                opacity=0.75,
                            ),
                        ),
                    },
                )
                self.region_safe_points_marker = VisualizationMarkers(cfg)

            if (
                getattr(self, "region_box_vis_translations", None) is not None
                and getattr(self, "region_box_vis_scales", None) is not None
                and not hasattr(self, "region_box_marker")
            ):
                cfg = CUBOID_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/region_boxes"
                cfg.markers["cuboid"].size = (1.0, 1.0, 1.0)
                cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.1, 0.8)
                cfg.markers["cuboid"].visual_material.opacity = 0.45
                self.region_box_marker = VisualizationMarkers(cfg)

            if not hasattr(self, "guidance_path_marker"):
                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/guidance_path",
                    markers={
                        "point": sim_utils.SphereCfg(
                            radius=0.06,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.15, 0.95, 0.25),
                                opacity=0.85,
                            ),
                        ),
                    },
                )
                self.guidance_path_marker = VisualizationMarkers(cfg)

            self.goal_marker.set_visibility(True)
            self.spawn_marker.set_visibility(True)
            self.desired_velocity_marker.set_visibility(True)
            self.current_velocity_marker.set_visibility(True)
            if hasattr(self, "region_safe_points_marker"):
                self.region_safe_points_marker.set_visibility(True)
            if hasattr(self, "region_box_marker"):
                self.region_box_marker.set_visibility(True)
            if hasattr(self, "guidance_path_marker"):
                self.guidance_path_marker.set_visibility(True)
        else:
            for name in [
                "goal_marker",
                "spawn_marker",
                "desired_velocity_marker",
                "current_velocity_marker",
                "region_safe_points_marker",
                "region_box_marker",
                "guidance_path_marker",
            ]:
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_marker.visualize(self.goal_position_world)
        self.spawn_marker.visualize(self.spawn_position_world)
        if hasattr(self, "region_safe_points_marker") and self.region_safe_points_vis is not None:
            self.region_safe_points_marker.visualize(
                translations=self.region_safe_points_vis,
                marker_indices=self.region_safe_points_vis_indices,
            )
        if (
            hasattr(self, "region_box_marker")
            and self.region_box_vis_translations is not None
            and self.region_box_vis_scales is not None
        ):
            self.region_box_marker.visualize(
                translations=self.region_box_vis_translations,
                scales=self.region_box_vis_scales,
            )
        if hasattr(self, "guidance_path_marker"):
            guidance_vis_points = self._get_guidance_vis_points_for_env(env_index=0)
            if guidance_vis_points is not None:
                self.guidance_path_marker.visualize(translations=guidance_vis_points)

        arrow_position = self.robot.data.root_pos_w.clone()
        arrow_position[:, 2] += 0.3

        desired_scale, desired_quat = self._compute_velocity_arrow(self.command[:, :3], is_goal_direction=True)
        self.desired_velocity_marker.visualize(arrow_position, desired_quat, desired_scale)

        current_scale, current_quat = self._compute_velocity_arrow(self.robot.data.root_lin_vel_b, is_goal_direction=False)
        self.current_velocity_marker.visualize(arrow_position, current_quat, current_scale)

    def _compute_velocity_arrow(self, velocity: torch.Tensor, is_goal_direction: bool):
        base_scale = torch.tensor(self.desired_velocity_marker.cfg.markers["arrow"].scale, device=self.device).repeat(
            velocity.shape[0], 1
        )
        if not is_goal_direction:
            velocity = velocity.clone()
            velocity[:, 2] = 0.0

        base_scale[:, 0] *= torch.norm(velocity, dim=1) * 3.0
        quat = vec_to_quat(velocity)

        if is_goal_direction:
            quat = math_utils.quat_mul(self.robot.data.root_quat_w, quat)
        else:
            quat = math_utils.quat_mul(yaw_quat(self.robot.data.root_quat_w), quat)
        return base_scale, quat


StaticRegionGoalCommand.pos_command_b = property(lambda self: self.goal_command_body)
StaticRegionGoalCommand.pos_command_w = property(lambda self: self.goal_position_world)
StaticRegionGoalCommand.pos_spawn_w = property(lambda self: self.spawn_position_world)
StaticRegionGoalCommand.closes_distance_to_goal = property(lambda self: self.closest_distance_to_goal)
StaticRegionGoalCommand.time_at_goal_in_steps = property(lambda self: self.steps_at_goal)
StaticRegionGoalCommand.required_time_at_goal_in_steps = property(lambda self: self.required_steps_at_goal)
StaticRegionGoalCommand.goal_reached_buffer = property(lambda self: self.success_tracker)
StaticRegionGoalCommand.goal_reached_counter = property(lambda self: self.goal_reach_count)
StaticRegionGoalCommand.distance_traveled = property(lambda self: self.total_distance_traveled)
StaticRegionGoalCommand.previous_pos_3d = property(lambda self: self.previous_position)

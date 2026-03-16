"""Goal command generator for region-based navigation on a static scan mesh."""

from __future__ import annotations

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
_REGION_POINT_SET_CACHE: dict[tuple[str, str, float, float, int, float], tuple[list[dict], np.ndarray]] = {}


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


def _estimate_region_ceiling_z(
    mesh: trimesh.Trimesh,
    region: dict,
    flight_height: float,
    min_room_height: float = 0.5,
) -> float:
    xy_min = np.asarray(region["xy_min"], dtype=np.float32)
    xy_max = np.asarray(region["xy_max"], dtype=np.float32)
    center_xy = np.asarray(region["center_xy"], dtype=np.float32)
    floor_z = float(region["floor_z"])

    span = np.maximum(xy_max - xy_min, 1e-3)
    offsets = np.array(
        [[0.0, 0.0], [-0.45, 0.0], [0.45, 0.0], [0.0, -0.45], [0.0, 0.45], [-0.3, -0.3], [-0.3, 0.3], [0.3, -0.3], [0.3, 0.3]],
        dtype=np.float32,
    )
    sample_xy = np.clip(center_xy + offsets * (span * 0.5), xy_min + 0.05, xy_max - 0.05)
    ray_origins = np.column_stack(
        (sample_xy[:, 0], sample_xy[:, 1], np.full(len(sample_xy), max(flight_height, floor_z + 0.05), dtype=np.float32))
    )
    ray_directions = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(sample_xy), 1))
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
    )

    candidate_ceilings: list[float] = []
    for ray_id in range(len(sample_xy)):
        ray_hits = locations[index_ray == ray_id]
        valid_hits = ray_hits[ray_hits[:, 2] > floor_z + min_room_height]
        if len(valid_hits) > 0:
            candidate_ceilings.append(float(valid_hits[:, 2].min()))

    if candidate_ceilings:
        return float(np.percentile(np.asarray(candidate_ceilings, dtype=np.float32), 75.0))
    return float(flight_height + 2.0)


def _build_occupied_kdtree(mesh: trimesh.Trimesh, num_surface_samples: int = 150000) -> KDTree:
    try:
        surface_points, _ = trimesh.sample.sample_surface_even(mesh, num_surface_samples)
    except Exception:
        surface_points = np.asarray(mesh.vertices, dtype=np.float32)
    return KDTree(np.asarray(surface_points, dtype=np.float32))


def _sample_region_safe_points(
    region: dict,
    occupied_kdtree: KDTree,
    flight_height: float,
    point_clearance: float,
    target_points_per_region: int,
    center_bias_ratio: float,
) -> np.ndarray:
    xy_min = np.asarray(region["xy_min"], dtype=np.float32)
    xy_max = np.asarray(region["xy_max"], dtype=np.float32)
    center_xy = np.asarray(region["center_xy"], dtype=np.float32)
    floor_z = float(region["floor_z"])
    ceiling_z = float(region["ceiling_z"])

    if not (floor_z + point_clearance <= flight_height <= ceiling_z - point_clearance):
        return np.zeros((0, 3), dtype=np.float32)

    shrunk_xy_min = xy_min + point_clearance
    shrunk_xy_max = xy_max - point_clearance
    if np.any(shrunk_xy_max <= shrunk_xy_min):
        shrunk_xy_min = xy_min.copy()
        shrunk_xy_max = xy_max.copy()

    span_xy = np.maximum(shrunk_xy_max - shrunk_xy_min, 1e-3)
    num_candidates = max(target_points_per_region * 16, 512)
    center_bias_ratio = float(np.clip(center_bias_ratio, 0.0, 1.0))
    center_bias_count = int(round(num_candidates * center_bias_ratio))
    uniform_count = num_candidates - center_bias_count

    gaussian_xy = np.zeros((0, 2), dtype=np.float32)
    if center_bias_count > 0:
        gaussian_xy = np.random.normal(
            loc=center_xy,
            scale=np.maximum(span_xy * 0.18, 0.08),
            size=(center_bias_count, 2),
        ).astype(np.float32)
        gaussian_xy = np.clip(gaussian_xy, shrunk_xy_min, shrunk_xy_max)

    uniform_xy = np.zeros((0, 2), dtype=np.float32)
    if uniform_count > 0:
        uniform_xy = np.random.uniform(low=shrunk_xy_min, high=shrunk_xy_max, size=(uniform_count, 2)).astype(np.float32)

    candidate_xy = np.concatenate((gaussian_xy, uniform_xy), axis=0)
    candidate_points = np.column_stack(
        (
            candidate_xy[:, 0],
            candidate_xy[:, 1],
            np.full(len(candidate_xy), flight_height, dtype=np.float32),
        )
    )

    distances, _ = occupied_kdtree.query(candidate_points, k=1)
    valid_points = candidate_points[np.asarray(distances) >= point_clearance]
    if len(valid_points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    distances_to_center = np.linalg.norm(valid_points[:, :2] - center_xy[None, :], axis=1)
    keep_count = min(len(valid_points), max(target_points_per_region, 64))
    return valid_points[np.argsort(distances_to_center)[:keep_count]].astype(np.float32, copy=False)


def _build_region_point_sets(
    surface_bbox_data_path: str,
    map_mesh_prim_path: str,
    flight_height: float,
    point_clearance: float,
    region_points_per_region: int,
    region_center_bias_ratio: float,
) -> tuple[list[dict], np.ndarray]:
    cache_key = (
        surface_bbox_data_path,
        map_mesh_prim_path,
        float(flight_height),
        float(point_clearance),
        int(region_points_per_region),
        float(region_center_bias_ratio),
    )
    if cache_key in _REGION_POINT_SET_CACHE:
        return _REGION_POINT_SET_CACHE[cache_key]

    mesh = _trimesh_from_stage_mesh(map_mesh_prim_path)
    occupied_kdtree = _build_occupied_kdtree(mesh)
    regions = _load_region_boxes(surface_bbox_data_path)

    valid_region_sets: list[dict] = []
    for region in regions:
        region["ceiling_z"] = _estimate_region_ceiling_z(mesh, region, flight_height=flight_height)
        points = _sample_region_safe_points(
            region=region,
            occupied_kdtree=occupied_kdtree,
            flight_height=flight_height,
            point_clearance=point_clearance,
            target_points_per_region=region_points_per_region,
            center_bias_ratio=region_center_bias_ratio,
        )
        if len(points) == 0:
            continue
        valid_region_sets.append(
            {
                "name": region["name"],
                "xy_min": region["xy_min"],
                "xy_max": region["xy_max"],
                "center": np.array([region["center_xy"][0], region["center_xy"][1], flight_height], dtype=np.float32),
                "spawn_points": points,
                "target_points": points.copy(),
            }
        )

    if len(valid_region_sets) < 2:
        raise RuntimeError(
            "Expected at least two valid region point sets from "
            f"{surface_bbox_data_path}, but got {len(valid_region_sets)}."
        )

    centers = np.asarray([region["center"] for region in valid_region_sets], dtype=np.float32)
    _REGION_POINT_SET_CACHE[cache_key] = (valid_region_sets, centers)
    return valid_region_sets, centers


class StaticRegionGoalCommand(CommandTerm):
    """Sample spawn and goal points from predefined mesh regions."""

    cfg: StaticRegionGoalCommandCfg

    def __init__(self, cfg: StaticRegionGoalCommandCfg, env: ManagerBasedRLEnv):
        self.visualize_region_safe_points = bool(cfg.visualize_region_safe_points)
        self.region_safe_points_vis: torch.Tensor | None = None
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        region_point_sets, region_centers = _build_region_point_sets(
            surface_bbox_data_path=cfg.surface_bbox_data_path,
            map_mesh_prim_path=cfg.map_mesh_prim_path,
            flight_height=float(cfg.flight_height),
            point_clearance=float(cfg.point_clearance),
            region_points_per_region=int(cfg.region_points_per_region),
            region_center_bias_ratio=float(cfg.region_center_bias_ratio),
        )
        all_regions = _load_region_boxes(cfg.surface_bbox_data_path)
        self.region_names = [region["name"] for region in region_point_sets]
        self.region_xy_min = torch.as_tensor(
            np.asarray([region["xy_min"] for region in region_point_sets], dtype=np.float32), device=self.device
        )
        self.region_xy_max = torch.as_tensor(
            np.asarray([region["xy_max"] for region in region_point_sets], dtype=np.float32), device=self.device
        )
        self.region_centers = torch.as_tensor(region_centers[:, :2], device=self.device)
        self.region_spawn_point_sets = [
            torch.as_tensor(region["spawn_points"], dtype=torch.float32, device=self.device) for region in region_point_sets
        ]
        self.region_target_point_sets = [
            torch.as_tensor(region["target_points"], dtype=torch.float32, device=self.device) for region in region_point_sets
        ]
        self.flight_height = float(cfg.flight_height)
        self.visualize_region_boxes = bool(cfg.visualize_region_boxes)
        self.region_safe_points_vis, self.region_safe_points_vis_indices = self._build_region_safe_points_vis(
            all_regions=all_regions,
            valid_regions=region_point_sets,
            points_per_region=int(cfg.region_safe_points_vis_points_per_region),
        )
        self.region_box_vis_translations, self.region_box_vis_scales = self._build_region_box_vis(
            all_regions=all_regions,
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

    def _build_region_safe_points_vis(
        self,
        all_regions: list[dict],
        valid_regions: list[dict],
        points_per_region: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.visualize_region_safe_points:
            return None, None

        valid_regions_by_name = {region["name"]: region for region in valid_regions}
        vis_points: list[torch.Tensor] = []
        vis_indices: list[torch.Tensor] = []
        for region in all_regions:
            valid_region = valid_regions_by_name.get(region["name"])
            if valid_region is not None and len(valid_region["spawn_points"]) > 0:
                points = torch.as_tensor(valid_region["spawn_points"], dtype=torch.float32, device=self.device)
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

    def _sample_points_from_region_sets(self, region_ids: torch.Tensor, region_point_sets: list[torch.Tensor]) -> torch.Tensor:
        sampled_points = torch.zeros((len(region_ids), 3), dtype=torch.float32, device=self.device)
        for region_id in torch.unique(region_ids):
            region_id_int = int(region_id.item())
            mask = region_ids == region_id
            candidates = region_point_sets[region_id_int]
            indices = torch.randint(0, len(candidates), (int(mask.sum().item()),), device=self.device)
            sampled_points[mask] = candidates[indices]
        return sampled_points

    def _sample_region_pairs(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_regions = len(self.region_spawn_point_sets)
        spawn_region_ids = torch.randint(0, num_regions, (len(env_ids),), device=self.device)
        goal_region_ids = torch.randint(0, num_regions - 1, (len(env_ids),), device=self.device)
        goal_region_ids += (goal_region_ids >= spawn_region_ids).long()
        spawn_points = self._sample_points_from_region_sets(spawn_region_ids, self.region_spawn_point_sets)
        goal_points = self._sample_points_from_region_sets(goal_region_ids, self.region_target_point_sets)

        return spawn_region_ids, goal_region_ids, spawn_points, goal_points

    def _update_metrics(self):
        position_error = self.goal_position_world - self.robot.data.root_pos_w[:, :3]
        position_error_2d = position_error[:, :2]
        velocity_2d = self.robot.data.root_state_w[:, 7:9]

        self.metrics["velocity_magnitude"] = torch.norm(velocity_2d, dim=1)
        direction_to_goal = position_error_2d / torch.clamp(torch.norm(position_error_2d, dim=1, keepdim=True), min=1e-6)
        self.metrics["velocity_toward_goal"] = (velocity_2d * direction_to_goal).sum(dim=1)
        self.metrics["success_rate"] = self.success_tracker.get_success_rate()

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.steps_at_goal[env_ids_tensor] = 0
        self.time_at_goal[env_ids_tensor] = 0
        self.total_distance_traveled[env_ids_tensor] = 0.0

        spawn_region_ids, goal_region_ids, spawn_points, goal_points = self._sample_region_pairs(env_ids_tensor)
        self.spawn_region_ids[env_ids_tensor] = spawn_region_ids
        self.goal_region_ids[env_ids_tensor] = goal_region_ids
        self.spawn_position_world[env_ids_tensor] = spawn_points
        self.goal_position_world[env_ids_tensor] = goal_points
        self.env.scene.env_origins[env_ids_tensor] = spawn_points
        self.previous_position[env_ids_tensor] = spawn_points
        self.initial_distance_to_goal[env_ids_tensor] = torch.norm(goal_points - spawn_points, dim=1)
        self.closest_distance_to_goal[env_ids_tensor] = self.initial_distance_to_goal[env_ids_tensor]

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

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        metrics_obs = self.env.observation_manager.compute_group(group_name="metrics")
        if env_ids is None:
            env_ids = slice(None)

        success = metrics_obs["in_goal"][env_ids]
        failed = ~success
        self.success_rate_buffer[env_ids] = torch.roll(self.success_rate_buffer[env_ids], 1, dims=1)
        rate = success.float() - failed.float()
        rate[rate == 0] = -1
        self.success_rate_buffer[env_ids, 0] = rate

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

            self.goal_marker.set_visibility(True)
            self.spawn_marker.set_visibility(True)
            self.desired_velocity_marker.set_visibility(True)
            self.current_velocity_marker.set_visibility(True)
            if hasattr(self, "region_safe_points_marker"):
                self.region_safe_points_marker.set_visibility(True)
            if hasattr(self, "region_box_marker"):
                self.region_box_marker.set_visibility(True)
        else:
            for name in [
                "goal_marker",
                "spawn_marker",
                "desired_velocity_marker",
                "current_velocity_marker",
                "region_safe_points_marker",
                "region_box_marker",
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

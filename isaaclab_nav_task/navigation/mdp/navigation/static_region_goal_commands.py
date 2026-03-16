"""Goal command generator for region-based navigation on a static scan mesh."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms, transform_points, yaw_quat

from isaaclab_nav_task.navigation.mdp.math_utils import vec_to_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .static_region_goal_commands_cfg import StaticRegionGoalCommandCfg


_RECT_NAME_RE = re.compile(r"^Rectangle:\s*(.+)$")
_CORNER_RE = re.compile(r"Corner\s+\d+:\s+X:\s*([-+0-9.]+),\s*Y:\s*([-+0-9.]+),\s*Z:\s*([-+0-9.]+)")


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


def _load_region_boxes(path: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    names: list[str] = []
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
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
                    names.append(current_name)
                    mins.append(points.min(dim=0).values.tolist())
                    maxs.append(points.max(dim=0).values.tolist())
                current_name = match.group(1)
                current_points = []
                continue

            match = _CORNER_RE.search(line)
            if match is not None:
                current_points.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])

    if current_name is not None and current_points:
        points = torch.tensor(current_points, dtype=torch.float32)
        names.append(current_name)
        mins.append(points.min(dim=0).values.tolist())
        maxs.append(points.max(dim=0).values.tolist())

    if not names:
        raise ValueError(f"No region rectangles found in {path}")

    return names, torch.tensor(mins, dtype=torch.float32), torch.tensor(maxs, dtype=torch.float32)


class StaticRegionGoalCommand(CommandTerm):
    """Sample spawn and goal points from predefined mesh regions."""

    cfg: StaticRegionGoalCommandCfg

    def __init__(self, cfg: StaticRegionGoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        region_names, region_min, region_max = _load_region_boxes(cfg.surface_bbox_data_path)
        self.region_names = region_names
        self.region_xy_min = region_min[:, :2].to(self.device)
        self.region_xy_max = region_max[:, :2].to(self.device)
        self.region_centers = (self.region_xy_min + self.region_xy_max) * 0.5
        self.flight_height = float(cfg.flight_height)
        self.min_goal_distance = float(cfg.min_goal_distance)
        self.max_goal_distance = cfg.max_goal_distance

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

    @property
    def command(self) -> torch.Tensor:
        return self.goal_command_body

    def _get_unscaled_command(self) -> torch.Tensor:
        return self.goal_command_body_unscaled

    def _sample_point_in_region(self, region_ids: torch.Tensor) -> torch.Tensor:
        low = self.region_xy_min[region_ids]
        high = self.region_xy_max[region_ids]
        xy = low + torch.rand((len(region_ids), 2), device=self.device) * (high - low)
        z = torch.full((len(region_ids), 1), self.flight_height, device=self.device)
        return torch.cat((xy, z), dim=1)

    def _sample_region_pairs(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_regions = self.region_xy_min.shape[0]
        spawn_region_ids = torch.randint(0, num_regions, (len(env_ids),), device=self.device)
        goal_region_ids = torch.randint(0, num_regions - 1, (len(env_ids),), device=self.device)
        goal_region_ids += (goal_region_ids >= spawn_region_ids).long()

        spawn_points = self._sample_point_in_region(spawn_region_ids)
        goal_points = self._sample_point_in_region(goal_region_ids)

        for _ in range(8):
            distances = torch.norm(goal_points[:, :2] - spawn_points[:, :2], dim=1)
            too_close = distances < self.min_goal_distance
            if self.max_goal_distance is not None:
                too_far = distances > self.max_goal_distance
                invalid = torch.logical_or(too_close, too_far)
            else:
                invalid = too_close
            if not invalid.any():
                break
            goal_region_ids[invalid] = torch.randint(0, num_regions - 1, (invalid.sum(),), device=self.device)
            goal_region_ids[invalid] += (goal_region_ids[invalid] >= spawn_region_ids[invalid]).long()
            goal_points[invalid] = self._sample_point_in_region(goal_region_ids[invalid])

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

            self.goal_marker.set_visibility(True)
            self.spawn_marker.set_visibility(True)
            self.desired_velocity_marker.set_visibility(True)
            self.current_velocity_marker.set_visibility(True)
        else:
            for name in ["goal_marker", "spawn_marker", "desired_velocity_marker", "current_velocity_marker"]:
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_marker.visualize(self.goal_position_world)
        self.spawn_marker.visualize(self.spawn_position_world)

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

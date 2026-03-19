from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .static_region_goal_commands import StaticRegionGoalCommand


@configclass
class StaticRegionGoalCommandCfg(CommandTermCfg):
    """Command configuration for static-mesh region navigation."""

    class_type: type = StaticRegionGoalCommand

    asset_name: str = MISSING
    surface_bbox_data_path: str = MISSING
    map_mesh_prim_path: str = "/World/MapMesh"
    spawn_polygon_csv_path: str | None = None
    guidance_paths_data_path: str | None = None
    guidance_trajectories_data_path: str | None = None
    precomputed_safe_points_path: str | None = None
    guidance_trajectory_eval_dt: float = 0.05
    guidance_arc_length_spacing: float = 0.2
    flight_height: float = 1.2
    point_clearance: float = 0.15
    safe_point_grid_spacing: float = 0.25
    region_points_per_region: int = 192
    region_center_bias_ratio: float = 0.7
    visualize_region_safe_points: bool = False
    region_safe_points_vis_points_per_region: int = 50
    visualize_region_boxes: bool = False
    region_box_vis_height: float = 0.05
    region_vis_z_offset: float = 0.8
    min_goal_distance: float = 3.0
    max_goal_distance: float | None = None
    robot_to_goal_line_vis: bool = True
    goal_hold_time_initial_s: float = 1.0
    goal_hold_time_final_s: float = 4.0
    goal_hold_curriculum_steps: int = 3000

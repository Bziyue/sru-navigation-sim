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
    spawn_polygon_csv_path: str | None = None
    guidance_paths_data_path: str | None = None
    flight_height: float = 1.2
    min_goal_distance: float = 3.0
    max_goal_distance: float | None = None
    robot_to_goal_line_vis: bool = True

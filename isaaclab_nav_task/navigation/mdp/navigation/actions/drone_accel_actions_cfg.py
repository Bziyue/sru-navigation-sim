from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .drone_accel_actions import DroneAccelAction


@configclass
class DroneAccelActionCfg(ActionTermCfg):
    """Configuration for planar drone acceleration commands."""

    class_type: type[ActionTerm] = DroneAccelAction

    asset_name: str = MISSING
    scale: list[float] = [1.0, 1.0, 1.5, 0.15]
    offset: list[float] = [0.0, 0.0, 0.0, 0.0]
    use_raw_actions: bool = False
    policy_distr_type: str = "gaussian"
    enable_height_command: bool = False
    target_height: float = 1.2
    min_height: float = 0.5
    max_height: float = 2.0
    body_name: str = "body"
    max_acceleration: float = 1.0
    max_speed: float = 1.0
    use_controller: bool = False
    controller_decimation: int = 2
    controller_k_max_ang: float = 30.0

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
    scale: list[float] = [2.0, 2.0, 1.5]
    offset: list[float] = [0.0, 0.0, 0.0]
    use_raw_actions: bool = False
    policy_distr_type: str = "gaussian"
    target_height: float = 1.2
    body_name: str = "body"
    max_speed: float = 2.5
    use_controller: bool = False
    controller_decimation: int = 2
    controller_k_max_ang: float = 30.0

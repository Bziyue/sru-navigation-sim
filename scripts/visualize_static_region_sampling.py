#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

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

parser = argparse.ArgumentParser(description="Visualize all legal static-region spawn/goal points and region boxes.")
parser.add_argument("--task", type=str, default="Isaac-Nav-PPO-Drone-Static-Play-v0", help="Static drone task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--auto_reset_interval",
    type=int,
    default=0,
    help="If > 0, force an environment reset every N simulation steps to resample the active spawn/goal markers.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
import isaaclab_nav_task.navigation.config.drone  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.robot_goal.debug_vis = True
    env_cfg.commands.robot_goal.visualize_region_safe_points = True
    env_cfg.commands.robot_goal.region_safe_points_vis_points_per_region = 0
    env_cfg.commands.robot_goal.visualize_region_boxes = True

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    action_dim = base_env.action_manager.get_term("velocity_command").action_dim
    zero_actions = torch.zeros((base_env.num_envs, action_dim), dtype=torch.float32, device=base_env.device)

    env.reset()
    command_term = base_env.command_manager.get_term("robot_goal")
    total_safe_points = int(sum(len(region_points) for region_points in command_term.region_safe_points))
    print(
        "[INFO] Visualizing static-region sampling: "
        f"task={args_cli.task}, num_envs={base_env.num_envs}, regions={len(command_term.region_safe_points)}, "
        f"safe_points={total_safe_points}, flight_height={float(command_term.flight_height):.2f}"
    )
    if args_cli.auto_reset_interval > 0:
        print(f"[INFO] Auto-reset interval: {args_cli.auto_reset_interval} steps")

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(zero_actions)
        step_count += 1
        if args_cli.auto_reset_interval > 0 and step_count % args_cli.auto_reset_interval == 0:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

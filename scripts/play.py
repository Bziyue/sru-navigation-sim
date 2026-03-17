#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy (PPO/MDPO) with automatic checkpoint loading.

Usage:
    python scripts/play.py --task <task_name> [options]

Arguments:
    --task                   Task name (required, typically *-Play-v0 variant)
    --checkpoint             Path to model checkpoint (.pt file)
    --use_last_checkpoint    Use latest checkpoint from logs (default behavior)
    --num_envs              Number of parallel environments
    --video                 Enable video recording
    --video_length          Video length in steps (default: 200)

Examples:
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0 --checkpoint path/to/model.pt
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0 --video --num_envs 16

Note: Automatically finds latest checkpoint if --checkpoint not specified.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def _prepend_workspace_packages():
    """Prefer the local workspace packages over stale site-packages installs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_root = os.path.dirname(script_dir)
    workspace_root = os.path.dirname(sim_root)
    learning_root = os.path.join(workspace_root, "sru-navigation-learning")

    for path in [learning_root, sim_root]:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_prepend_workspace_packages()

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras
args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import re
import torch

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
import isaaclab_nav_task.navigation.config.drone  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx


PLAY_STATUS_INTERVAL = 20
TRAIL_POINT_CAP = 400


def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    """Find the latest checkpoint file in the log directory.

    Args:
        log_path: Base log directory path
        checkpoint_pattern: Regex pattern for checkpoint files

    Returns:
        Path to the latest checkpoint file
    """
    # Find all run directories
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    # Sort to get latest run
    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    # Find checkpoint files
    checkpoint_files = []
    for f in os.listdir(run_path):
        if re.match(checkpoint_pattern, f):
            checkpoint_files.append(f)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    # Sort to get latest checkpoint
    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    latest_checkpoint = checkpoint_files[-1]

    return os.path.join(run_path, latest_checkpoint)


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str, load_optimizer: bool = True):
    """Load checkpoint with fallback for PyTorch compatibility issues.

    Args:
        runner: RSL-RL runner instance
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer state
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint to CPU first for compatibility
    loaded_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model state - handle both standard algorithms (PPO) and MDPO
    if runner.is_mdpo:
        # MDPO uses two actor-critics, load same state into both
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        # Standard algorithms use one actor-critic
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    # Load normalizers if using empirical normalization
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    # Load optimizer if requested
    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as JIT module to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Export using the module's export_jit method
    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print(f"[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as ONNX model to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Check if the module has export_onnx method
    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    # Export using the module's export_onnx method
    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print(f"[INFO] ONNX export complete!")


def split_actor_observations(observations, extras=None):
    """Handle legacy `(obs, extras)` and IsaacLab TensorDict observations."""
    if isinstance(observations, tuple) and len(observations) == 2 and extras is None:
        observations, extras = observations

    if hasattr(observations, "keys"):
        keys = set(observations.keys())
        if "policy" in keys:
            return observations["policy"]
        if "observations" in keys:
            obs_group = observations["observations"]
            if hasattr(obs_group, "get"):
                return obs_group.get("policy", obs_group)
            return obs_group

    return observations


def create_drone_debug_marker() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Play/drone_marker",
        markers={
            "disc": sim_utils.CylinderCfg(
                radius=0.45,
                height=0.03,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.05, 0.05),
                    opacity=0.8,
                ),
            ),
        },
    )
    marker = VisualizationMarkers(cfg)
    marker.set_visibility(True)
    return marker


def create_point_marker(
    prim_path: str,
    color: tuple[float, float, float],
    radius: float,
    opacity: float = 0.9,
) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "point": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    opacity=opacity,
                ),
            ),
        },
    )
    marker = VisualizationMarkers(cfg)
    marker.set_visibility(True)
    return marker


def get_robot_goal_term(env):
    return getattr(env.unwrapped.command_manager, "_terms", {}).get("robot_goal")


def get_current_episode_signature(env) -> tuple[int, int, int, float, float]:
    command_term = get_robot_goal_term(env)
    if command_term is None:
        return (-1, -1, -1, 0.0, 0.0)

    guidance_id = int(command_term.current_guidance_ids[0].detach().cpu().item())
    spawn_region = int(command_term.spawn_region_ids[0].detach().cpu().item())
    goal_region = int(command_term.goal_region_ids[0].detach().cpu().item())
    goal = command_term.goal_position_world[0].detach().cpu()
    return (guidance_id, spawn_region, goal_region, round(float(goal[0]), 3), round(float(goal[1]), 3))


def get_guidance_path_points(env, max_points: int = 240) -> torch.Tensor | None:
    command_term = get_robot_goal_term(env)
    if command_term is None:
        return None

    guidance_id = int(command_term.current_guidance_ids[0].detach().cpu().item())
    path_length = int(command_term.guidance_path_lengths[guidance_id].detach().cpu().item())
    if path_length <= 0:
        return None

    path_xy = command_term.guidance_paths_xy[guidance_id, :path_length]
    if path_length > max_points:
        sample_idx = torch.linspace(0, path_length - 1, max_points, device=path_xy.device)
        path_xy = path_xy[sample_idx.round().long()]

    points = torch.zeros((len(path_xy), 3), dtype=torch.float32, device=path_xy.device)
    points[:, :2] = path_xy
    points[:, 2] = float(command_term.flight_height) + 0.06
    return points


def build_episode_static_points(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    command_term = get_robot_goal_term(env)
    if command_term is None:
        return None, None, None

    spawn_region = int(command_term.spawn_region_ids[0].detach().cpu().item())
    goal_region = int(command_term.goal_region_ids[0].detach().cpu().item())

    spawn_center = torch.zeros((1, 3), dtype=torch.float32, device=command_term.region_centers.device)
    spawn_center[0, :2] = command_term.region_centers[spawn_region]
    spawn_center[0, 2] = float(command_term.flight_height) + 0.08

    goal_center = torch.zeros((1, 3), dtype=torch.float32, device=command_term.region_centers.device)
    goal_center[0, :2] = command_term.region_centers[goal_region]
    goal_center[0, 2] = float(command_term.flight_height) + 0.08

    goal_point = command_term.goal_position_world[0:1].clone()
    goal_point[:, 2] += 0.10
    return spawn_center, goal_center, goal_point


def update_drone_debug_marker(marker: VisualizationMarkers, env):
    robot = env.unwrapped.scene["robot"]
    marker_pos = robot.data.root_pos_w[:, :3].clone()
    marker_pos[:, 2] -= 0.12
    marker.visualize(translations=marker_pos)


def update_trail_marker(marker: VisualizationMarkers, trail_points: list[torch.Tensor]):
    if not trail_points:
        return

    trail = torch.stack(trail_points, dim=0)
    if len(trail) > TRAIL_POINT_CAP:
        sample_idx = torch.linspace(0, len(trail) - 1, TRAIL_POINT_CAP, device=trail.device)
        trail = trail[sample_idx.round().long()]
    marker.visualize(translations=trail)


def print_play_status(env, step_count: int):
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    pos = robot.data.root_pos_w[0].detach().cpu()
    vel = robot.data.root_lin_vel_w[0].detach().cpu()
    yaw_rate = float(robot.data.root_ang_vel_b[0, 2].detach().cpu().item())

    status = (
        f"[PLAY step={step_count}] "
        f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
        f"vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) "
        f"yaw_rate={yaw_rate:.2f}"
    )

    command_term = getattr(base_env.command_manager, "_terms", {}).get("robot_goal")
    if command_term is not None:
        goal = command_term.goal_position_world[0].detach().cpu()
        distance_to_goal = float(command_term.distance_to_goal[0].detach().cpu().item())
        guidance_progress = float(command_term.guidance_progress[0].detach().cpu().item())
        guidance_lateral_error = float(command_term.guidance_lateral_error[0].detach().cpu().item())
        spawn_region = int(command_term.spawn_region_ids[0].detach().cpu().item())
        goal_region = int(command_term.goal_region_ids[0].detach().cpu().item())
        status += (
            f" goal=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}) "
            f"dist={distance_to_goal:.2f} "
            f"guidance_s={guidance_progress:.2f} "
            f"guidance_lat={guidance_lateral_error:.2f} "
            f"regions={spawn_region}->{goal_region}"
        )

    print(status, flush=True)


def main():
    """Play navigation policy with RSL-RL."""
    # Parse command-line arguments
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    # Instantiate the configs
    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    # Override config from command line
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap the environment
    env = RslRlVecEnvWrapper(env)

    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        # Get last checkpoint from log directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint with compatibility handling
    load_checkpoint_with_fallback(runner, resume_path)

    # Export JIT if requested
    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)

    # Export ONNX if requested
    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    # Obtain policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    drone_marker = create_drone_debug_marker()
    trail_marker = create_point_marker(
        prim_path="/Visuals/Play/drone_trail",
        color=(1.0, 0.3, 0.1),
        radius=0.045,
        opacity=0.85,
    )
    spawn_center_marker = create_point_marker(
        prim_path="/Visuals/Play/spawn_region_center",
        color=(0.1, 0.8, 1.0),
        radius=0.14,
        opacity=0.85,
    )
    goal_center_marker = create_point_marker(
        prim_path="/Visuals/Play/goal_region_center",
        color=(1.0, 0.85, 0.1),
        radius=0.14,
        opacity=0.85,
    )
    goal_point_marker = create_point_marker(
        prim_path="/Visuals/Play/goal_point",
        color=(1.0, 0.0, 1.0),
        radius=0.10,
        opacity=0.9,
    )
    guidance_path_marker = create_point_marker(
        prim_path="/Visuals/Play/guidance_path",
        color=(0.15, 0.95, 0.25),
        radius=0.05,
        opacity=0.8,
    )

    # Reset environment
    obs = split_actor_observations(env.get_observations())
    update_drone_debug_marker(drone_marker, env)
    episode_signature = get_current_episode_signature(env)
    current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :3].clone()
    current_pos[2] -= 0.02
    trail_points = [current_pos]
    update_trail_marker(trail_marker, trail_points)
    spawn_center, goal_center, goal_point = build_episode_static_points(env)
    if spawn_center is not None:
        spawn_center_marker.visualize(translations=spawn_center)
    if goal_center is not None:
        goal_center_marker.visualize(translations=goal_center)
    if goal_point is not None:
        goal_point_marker.visualize(translations=goal_point)
    guidance_points = get_guidance_path_points(env)
    if guidance_points is not None:
        guidance_path_marker.visualize(translations=guidance_points)
    print_play_status(env, step_count=0)

    # Simulate environment
    step_count = 0
    while simulation_app.is_running():
        # Run policy
        with torch.inference_mode():
            actions = policy(obs)
        # Step environment
        obs_data, _, _, infos = env.step(actions)
        obs = split_actor_observations(obs_data, infos)
        step_count += 1
        update_drone_debug_marker(drone_marker, env)
        new_signature = get_current_episode_signature(env)
        if new_signature != episode_signature:
            episode_signature = new_signature
            current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :3].clone()
            current_pos[2] -= 0.02
            trail_points = [current_pos]
            spawn_center, goal_center, goal_point = build_episode_static_points(env)
            if spawn_center is not None:
                spawn_center_marker.visualize(translations=spawn_center)
            if goal_center is not None:
                goal_center_marker.visualize(translations=goal_center)
            if goal_point is not None:
                goal_point_marker.visualize(translations=goal_point)
            guidance_points = get_guidance_path_points(env)
            if guidance_points is not None:
                guidance_path_marker.visualize(translations=guidance_points)
        else:
            current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :3].clone()
            current_pos[2] -= 0.02
            trail_points.append(current_pos)

        update_trail_marker(trail_marker, trail_points)
        if step_count % PLAY_STATUS_INTERVAL == 0:
            print_play_status(env, step_count)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()

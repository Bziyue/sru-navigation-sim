import gymnasium as gym

from . import agents, swarm_navigation_env_cfg


gym.register(
    id="Isaac-Nav-PPO-Drone-Swarm-Static-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmStaticNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": swarm_navigation_env_cfg.DroneSwarmStaticNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmStaticNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Drone-Swarm-Static-Play-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmStaticNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": swarm_navigation_env_cfg.DroneSwarmStaticNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmStaticNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Drone-Swarm-Static-Dev-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmStaticNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": swarm_navigation_env_cfg.DroneSwarmStaticNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmStaticNavPPORunnerDevCfg,
    },
)


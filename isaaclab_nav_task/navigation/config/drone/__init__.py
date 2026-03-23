import gymnasium as gym

from . import agents, navigation_env_cfg


gym.register(
    id="Isaac-Nav-PPO-Drone-Static-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DroneStaticNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneStaticNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Drone-Static-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DroneStaticNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneStaticNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Drone-Static-PlayFast-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DroneStaticNavigationEnvCfg_PLAY_FAST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneStaticNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Drone-Static-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DroneStaticNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneStaticNavPPORunnerDevCfg,
    },
)


gym.register(
    id="Isaac-Nav-PPO-Drone-Static-SwarmCompat-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DroneStaticNavigationEnvCfg_SWARM_COMPAT,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneStaticNavPPOCompatRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Drone-Static-SwarmCompat-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DroneStaticNavigationEnvCfg_SWARM_COMPAT_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneStaticNavPPOCompatRunnerDevCfg,
    },
)

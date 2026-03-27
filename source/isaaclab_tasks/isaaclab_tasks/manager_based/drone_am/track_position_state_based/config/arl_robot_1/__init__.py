# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 训练任务（完整配置）
gym.register(
    id="Isaac-TrackPositionNoObstacles-AM-Robot-1-v0",
    # 统一走 ManagerBasedRLEnv 构建流程
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口（场景/动作/观测/奖励/终止等）
        "env_cfg_entry_point": f"{__name__}.no_obstacle_env_cfg:NoObstacleEnvCfg",
        # 三套训练器配置入口：rl_games / rsl_rl / skrl
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrackPositionNoObstaclesEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# 演示/调试任务（PLAY 配置，环境规模更小、随机化更少）
gym.register(
    id="Isaac-TrackPositionNoObstacles-AM-Robot-1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_obstacle_env_cfg:NoObstacleEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrackPositionNoObstaclesEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

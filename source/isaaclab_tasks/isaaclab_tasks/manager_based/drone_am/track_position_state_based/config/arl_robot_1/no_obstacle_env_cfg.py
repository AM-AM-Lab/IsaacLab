# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_assets.robots.drone_am import DRONE_AM_CFG

from .track_position_state_based_env_cfg import TrackPositionNoObstaclesEnvCfg

##
# Pre-defined configs
##


@configclass
class NoObstacleEnvCfg(TrackPositionNoObstaclesEnvCfg):
    def __post_init__(self):
        # post init of parent
        # 先执行父类后处理：初始化通用场景/动作/观测/奖励等配置
        super().__post_init__()
        # switch robot to drone_am
        # 将通用模板中的占位机器人替换为 Drone AM (12-rotor hexacopter)
        self.scene.robot = DRONE_AM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 同步推进器内部时间步与仿真时间步，避免执行器离散化不一致
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt


@configclass
class NoObstacleEnvCfg_PLAY(NoObstacleEnvCfg):
    def __post_init__(self):
        # post init of parent
        # 复用训练配置，再覆盖为更轻量的可视化/调试配置
        super().__post_init__()

        # make a smaller scene for play
        # PLAY 模式减少并行环境数，便于本地查看与调试
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        # 关闭观测扰动，让现象更稳定、更易观察
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # 去掉外力扰动事件，避免演示时出现随机推搡
        self.events.base_external_force_torque = None
        self.events.push_robot = None

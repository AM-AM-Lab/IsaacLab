# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass

from .drone_pose_command import DroneUniformPoseCommand


@configclass
class DroneUniformPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for uniform drone pose command generator."""

    # 中文：将通用命令配置的执行类替换为 Drone 版本实现。
    # 这样在环境中声明 DroneUniformPoseCommandCfg 时，会实例化 DroneUniformPoseCommand。

    class_type: type = DroneUniformPoseCommand

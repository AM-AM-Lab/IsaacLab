# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error


class DroneUniformPoseCommand(UniformPoseCommand):
    """Drone-specific UniformPoseCommand extensions.

    This class customizes the generic :class:`UniformPoseCommand` for drone (multirotor)
    use-cases. Main differences and additions:

    - Transforms pose commands from the drone's base frame to the world frame before use.
    - Accounts for per-environment origin offsets (``scene.env_origins``) when computing
        position errors so tasks running on shifted/sub-terrain environments compute
        meaningful errors.
    - Computes and exposes simple metrics used by higher-level code: ``position_error``
        and ``orientation_error`` (stored in ``self.metrics``).
    - Provides a debug visualization callback that renders the goal pose (with
        sub-terrain shift) and current body pose using the existing visualizers.

    The implementation overrides :meth:`_update_metrics` and :meth:`_debug_vis_callback`
    from the base class to implement these drone-specific behaviors.
    """

    # 中文说明：
    # 该类在通用 UniformPoseCommand 基础上做了两件关键事：
    # 1) 计算误差时考虑并行环境原点偏移 env_origins；
    # 2) debug 可视化时同样加入 env_origins，确保标记与子环境对齐。

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        # 中文：把“机体系命令”转换到“世界系命令”，便于与仿真真值比较。
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            # Sub-terrain shift for correct position error calculation @grzemal
            # 中文：加上 env_origins，避免多环境平铺时误差计算错位。
            self.pose_command_b[:, :3] + self._env.scene.env_origins,
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        # 中文：记录位置/姿态误差范数，供日志与调试使用。
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        # Sub-terrain shift for visualization purposes @grzemal
        # 中文：目标可视化也要加 env_origins，才能与各子环境正确重合。
        self.goal_pose_visualizer.visualize(
            self.pose_command_b[:, :3] + self._env.scene.env_origins, self.pose_command_b[:, 3:]
        )
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])

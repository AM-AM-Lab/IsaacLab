# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_uniform_absolute(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset selected joints to absolute values uniformly sampled in the given ranges.

    Unlike ``reset_joints_by_offset``, this term does not depend on ``default_joint_pos``,
    which may be unavailable for multirotor assets with only thruster actuators.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve joint count for selected joints.
    if isinstance(asset_cfg.joint_ids, slice):
        joint_ids: list[int] | slice = slice(None)
        num_joints = asset.num_joints
    else:
        joint_ids = list(asset_cfg.joint_ids)
        num_joints = len(joint_ids)

    shape = (env_ids.shape[0], num_joints)
    joint_pos = sample_uniform(position_range[0], position_range[1], shape, asset.device)
    joint_vel = sample_uniform(velocity_range[0], velocity_range[1], shape, asset.device)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)  # type: ignore[arg-type]

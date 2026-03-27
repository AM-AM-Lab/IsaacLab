# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the drone AM environments."""

# 中文说明：
# 该文件是 drone_am 的“统一导出入口”。
# 在 env_cfg 中写 `import ... as mdp` 后，可直接通过 `mdp.xxx` 使用：
# - Isaac Lab 内置 MDP 项（isaaclab.envs.mdp）
# - contrib 扩展 MDP 项（isaaclab_contrib.mdp）
# - 本任务自定义项（commands / observations / rewards）
# 这样能把环境配置写得更简洁，且便于按需替换实现。

from isaaclab.envs.mdp import *  # noqa: F401, F403

from isaaclab_contrib.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

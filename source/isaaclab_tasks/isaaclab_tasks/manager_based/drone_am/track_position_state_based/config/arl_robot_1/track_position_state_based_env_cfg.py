# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_assets.robots.drone_am import DRONE_AM_CFG

import isaaclab_tasks.manager_based.drone_am.mdp as mdp


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    # robots
    robot: MultirotorCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            # 场景照明（仅影响视觉，不直接影响控制）
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 目标位姿命令（本配置当前范围全为 0，本质是定点悬停任务）
    target_pose = mdp.DroneUniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        # 每 10s 重采样一次目标；当前范围固定为常值，因此重采样结果不变
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.DroneUniformPoseCommandCfg.Ranges(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(1.2, 1.2),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 12桨推力动作：processed = raw * scale + offset，并做 clip
    thrust_command = mdp.ThrustActionCfg(
        asset_name="robot",
        scale=6.0,
        offset=20.0,
        preserve_order=False,
        use_default_offset=False,
        bind_joint_velocity_to_thrust=True,
        joint_names_expr=[
            "rotor_0_joint",
            "rotor_1_joint",
            "rotor_2_joint",
            "rotor_3_joint",
            "rotor_4_joint",
            "rotor_5_joint",
            "rotor_6_joint",
            "rotor_7_joint",
            "rotor_8_joint",
            "rotor_9_joint",
            "rotor_10_joint",
            "rotor_11_joint",
        ],
        joint_velocity_scale=100.0,
        joint_velocity_offset=0.0,
        use_rotor_directions_for_joint_velocity=True,
        write_joint_velocity_to_sim=True,
        clip={
            "rotor_0": (0.0, 25.0),
            "rotor_1": (0.0, 25.0),
            "rotor_2": (0.0, 25.0),
            "rotor_3": (0.0, 25.0),
            "rotor_4": (0.0, 25.0),
            "rotor_5": (0.0, 25.0),
            "rotor_6": (0.0, 25.0),
            "rotor_7": (0.0, 25.0),
            "rotor_8": (0.0, 25.0),
            "rotor_9": (0.0, 25.0),
            "rotor_10": (0.0, 25.0),
            "rotor_11": (0.0, 25.0),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 观测项（按定义顺序拼接）
        base_link_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 机械臂 6 关节角（rm_joint1~rm_joint6）
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rm_joint1", "rm_joint2", "rm_joint3", "rm_joint4", "rm_joint5", "rm_joint6"])} ,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        # 上一时刻动作，有助于抑制抖振和处理执行器滞后
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            # 关闭观测扰动（训练更稳定；需要鲁棒性时可开启）
            self.enable_corruption = False
            # 拼接为单一 policy 向量输入网络
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset

    reset_base = EventTerm(
        # 在每次 reset 时随机化初始位姿与速度，提升泛化
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.8, 1.5),
                # "yaw": (-math.pi / 6.0, math.pi / 6.0),
                # "roll": (-math.pi / 6.0, math.pi / 6.0),
                # "pitch": (-math.pi / 6.0, math.pi / 6.0),
                "yaw": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.05, 0.05),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # 机械臂关节 reset：除 z2 外其余关节固定为 0
    reset_arm_joints_rest = EventTerm(
        func=mdp.reset_joints_uniform_absolute,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["rm_joint1", "rm_joint3", "rm_joint4", "rm_joint5", "rm_joint6"],
            ),
        },
    )

    # z2 关节在 [45°, 100°] 内随机
    reset_arm_joint_z1 = EventTerm(
        func=mdp.reset_joints_uniform_absolute,
        mode="reset",
        params={
            "position_range": (math.radians(45.0), math.radians(100.0)),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["rm_joint2"]),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 主任务奖励：离目标越近越大（该项权重最高）
    distance_to_goal_exp = RewTerm(
        func=mdp.distance_to_goal_exp,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "target_pose",
        },
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 航向对齐奖励
    yaw_aligned = RewTerm(
        func=mdp.yaw_aligned,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    )
    # 速度稳定奖励（抑制漂移和剧烈旋转）
    lin_vel_xyz_exp = RewTerm(
        func=mdp.lin_vel_xyz_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
    )
    ang_vel_xyz_exp = RewTerm(
        func=mdp.ang_vel_xyz_exp,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 10.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)

    # 提前终止（如坠落）额外惩罚
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 到最大时长结束
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 高度低于阈值视为坠落
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.5})


##
# Environment configuration
##


@configclass
class TrackPositionNoObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the state-based drone pose-control environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # 环境步频 = 物理步频 / decimation
        self.decimation = 10
        self.episode_length_s = 10
        # 物理步长 0.01s（100Hz）
        self.sim.dt = 0.01
        # 渲染频率与环境步同步（每 decimation 个物理步渲染一次）
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        # 提高 GPU 刚体接触 patch 容量，减少大规模并行下接触相关报错
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

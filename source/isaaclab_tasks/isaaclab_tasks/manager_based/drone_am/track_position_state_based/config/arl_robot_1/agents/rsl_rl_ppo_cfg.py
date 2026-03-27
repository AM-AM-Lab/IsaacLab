# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TrackPositionNoObstaclesEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境每次 rollout 收集的步数
    num_steps_per_env = 24
    # 最大训练迭代次数
    max_iterations = 1500
    # 每隔多少迭代保存一次 checkpoint
    save_interval = 50
    experiment_name = "arl_robot_1_track_position_state_based"
    # 是否使用经验归一化（此处关闭）
    empirical_normalization = False

    # Actor-Critic 网络结构
    policy = RslRlPpoActorCriticCfg(
        # 初始策略探索噪声标准差
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # PPO 算法超参数
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值函数损失权重
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # PPO clip 系数
        clip_param=0.2,
        # 熵奖励系数（鼓励探索）
        entropy_coef=0.001,
        # 每次 rollout 的优化 epoch 与 mini-batch 切分
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-4,
        # 自适应学习率调度（依据 KL）
        schedule="adaptive",
        # 折扣因子与 GAE 参数
        gamma=0.98,
        lam=0.95,
        # 期望 KL 与梯度裁剪
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

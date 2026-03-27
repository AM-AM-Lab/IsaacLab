# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Drone AM (12-rotor hexacopter).

The following configuration parameters are available:

* :obj:`DRONE_AM_CFG`: The Drone AM with 12 rotors in coaxial hexacopter configuration.
"""

import isaaclab.sim as sim_utils
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

##
# Configuration - Actuators (12-rotor thruster).
##

DRONE_AM_THRUSTER = ThrusterCfg(
    thrust_range=(0.1, 30.0),
    thrust_const_range=(9.26312e-06, 1.826312e-05),
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.005, 0.005),
    torque_to_thrust_ratio=0.07,
    thruster_names_expr=[
        "rotor_0", "rotor_1", "rotor_2", "rotor_3",
        "rotor_4", "rotor_5", "rotor_6", "rotor_7",
        "rotor_8", "rotor_9", "rotor_10", "rotor_11",
    ],
)

##
# Configuration - Articulation (12-rotor hexacopter).
##

DRONE_AM_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/AM/am20/am20.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        rps={
            "rotor_0": 200.0,
            "rotor_1": 200.0,
            "rotor_2": 200.0,
            "rotor_3": 200.0,
            "rotor_4": 200.0,
            "rotor_5": 200.0,
            "rotor_6": 200.0,
            "rotor_7": 200.0,
            "rotor_8": 200.0,
            "rotor_9": 200.0,
            "rotor_10": 200.0,
            "rotor_11": 200.0,
        },
    ),
    actuators={"thrusters": DRONE_AM_THRUSTER},
    # Rotor rotation directions: 1=CCW, -1=CW
    # Order follows the user-defined DODECAHEXA-X numbering in the figure:
    # rotor_0..rotor_11 <-> motor 1..12
    # [1..12] directions = [CCW, CW, CW, CCW, CCW, CW, CW, CCW, CCW, CW, CW, CCW]
    rotor_directions=[1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    # Allocation matrix: 6 x 12
    # Maps 12 rotor thrusts to [Fx, Fy, Fz, Mx, My, Mz]
    # Geometry per user definition:
    # - Body axes: X forward, Y right, Z down
    # - Arm span: 674.5 mm (radius = 0.33725 m)
    # - Tilt angle: 30 deg
    # - Motor order: rotor_0..rotor_11 <-> motor 1..12
    allocation_matrix=[
        # Fx: 1/2&7/8&11/12&5/6 have +0.5*sin(30), 3/4&9/10 tilt backward around Y axis
        [0.25, 0.25, -0.5, -0.5, 0.25, 0.25, 0.25, 0.25, -0.5, -0.5, 0.25, 0.25],
        # Fy: 1/2&7/8 have -0.5*cos(30), 11/12&5/6 have +0.5*cos(30)
        [-0.433, -0.433, 0.0, 0.0, 0.433, 0.433, -0.433, -0.433, 0.0, 0.0, 0.433, 0.433],
        # Fz (force along body Z, body Z points down -> upward thrust is negative Z)
        [-0.866, -0.866, -0.866, -0.866, -0.866, -0.866, -0.866, -0.866, -0.866, -0.866, -0.866, -0.866],
        # Mx from r x F with FRD sign convention (radius = 0.33725 m)
        [
            -0.1461, -0.1461, -0.2920, -0.2920, -0.1461, -0.1461,
            0.1461, 0.1461, 0.2920, 0.2920, 0.1461, 0.1461,
        ],
        # My from r x F with FRD sign convention (radius = 0.33725 m)
        [
            0.2530, 0.2530, 0.0, 0.0, -0.2530, -0.2530,
            -0.2530, -0.2530, 0.0, 0.0, 0.2530, 0.2530,
        ],
        # Mz (torque around body Z axis) = K_m * thrust
        # Sign follows figure-defined rotor directions above (|K_m| = 0.05)
        [
            0.05, -0.05, -0.05, 0.05, 0.05, -0.05,
            -0.05, 0.05, 0.05, -0.05, -0.05, 0.05,
        ],
    ],
)

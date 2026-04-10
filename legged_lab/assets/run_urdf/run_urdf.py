"""自定义 Run URDF 机器人的配置文件。

USD 文件路径: legged_lab/assets/Run_URDF/urdf/Run_URDF_fixed_v2/Run_URDF_fixed_v2.usd

关节结构（共 15 个自由度）:
  腰部   : waist_yaw（腰部偏航）
  腿部   : hip_roll_{r/l}（髋关节侧滚）, hip_yaw_{r/l}（髋关节偏航）,
           hip_pitch_{r/l}（髋关节俯仰）, knee_pitch_{r/l}（膝关节俯仰）,
           ankle_pitch_{r/l}（踝关节俯仰）
  手臂   : shoulder_pitch_{r/l}（肩关节俯仰）, elbow_pitch_{r/l}（肘关节俯仰）
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

# ──────────────────────────────────────────────────────────────
# Run URDF 机器人完整关节配置
# ──────────────────────────────────────────────────────────────
RUN_URDF_CFG = ArticulationCfg(
    # ── 外观与物理属性 ──────────────────────────────────────────
    spawn=sim_utils.UsdFileCfg(
        # USD 模型文件路径（固定底座版本）
        usd_path=f"{ISAAC_ASSET_DIR}/Run_URDF/urdf/Run_URDF_fixed_v2/Run_URDF_fixed_v2.usd",
        # 启用接触传感器，用于足端接触检测
        activate_contact_sensors=True,
        # 刚体物理属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,        # 启用重力（True 则悬浮，调试用）
            retain_accelerations=False,   # 不保留加速度缓存，节省内存
            linear_damping=0.0,           # 线性阻尼（0 = 无空气阻力）
            angular_damping=0.0,          # 角阻尼（0 = 无旋转阻力）
            max_linear_velocity=1000.0,   # 最大线速度上限 (m/s)
            max_angular_velocity=1000.0,  # 最大角速度上限 (rad/s)
            max_depenetration_velocity=1.0,  # 碰撞穿透后的最大分离速度 (m/s)
        ),
        # 关节链（Articulation）根节点物理属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,       # 启用自碰撞检测，防止肢体互穿
            solver_position_iteration_count=4,  # 位置求解器迭代次数（越大越精确，越慢）
            solver_velocity_iteration_count=4,  # 速度求解器迭代次数
        ),
    ),

    # ── 初始状态 ────────────────────────────────────────────────
    init_state=ArticulationCfg.InitialStateCfg(
        # 机器人基座初始位置 (x, y, z)，z=0.80m 使机器人站立于地面
        pos=(0.0, 0.0, 0.80),
        # 各关节初始角度（单位：弧度），未列出的关节默认为 0
        joint_pos={
            # 腿部关节初始姿态（轻度屈膝站立）
            "hip_pitch_r": -0.20,   # 右髋俯仰：略微前倾
            "hip_pitch_l": -0.20,   # 左髋俯仰：略微前倾
            "knee_pitch_r":  0.42,  # 右膝弯曲
            "knee_pitch_l":  0.42,  # 左膝弯曲
            "ankle_pitch_r": -0.23, # 右踝补偿膝弯，保持脚掌水平
            "ankle_pitch_l": -0.23, # 左踝补偿膝弯，保持脚掌水平
            # 手臂关节初始姿态（自然下垂略弯）
            "shoulder_pitch_r": 0.35,  # 右肩前倾
            "shoulder_pitch_l": 0.35,  # 左肩前倾
            "elbow_pitch_r": 0.87,     # 右肘弯曲
            "elbow_pitch_l": 0.87,     # 左肘弯曲
        },
        # 所有关节初始速度均为 0（".*" 为正则通配符，匹配全部关节）
        joint_vel={".*": 0.0},
    ),

    # 关节位置软限位系数：实际可用范围 = URDF 限位 × 0.90，留出安全裕量
    soft_joint_pos_limit_factor=0.90,

    # ── 执行器（驱动器）配置 ────────────────────────────────────
    # 按功能分为三组：腿部、足部、手臂，分别设置不同的力矩/速度/刚度/阻尼参数
    actuators={
        # ── 腿部 + 腰部执行器 ───────────────────────────────────
        # 包含髋关节（偏航/侧滚/俯仰）、膝关节、腰部偏航，共 9 个关节
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_r",    # 右髋偏航
                "hip_yaw_l",    # 左髋偏航
                "hip_roll_r",   # 右髋侧滚
                "hip_roll_l",   # 左髋侧滚
                "hip_pitch_r",  # 右髋俯仰
                "hip_pitch_l",  # 左髋俯仰
                "knee_pitch_r", # 右膝俯仰
                "knee_pitch_l", # 左膝俯仰
                "waist_yaw",    # 腰部偏航
            ],
            # 各关节力矩上限（单位：N·m）
            # hip_roll / knee 承受更大载荷，力矩限位更高
            effort_limit_sim={
                "hip_yaw_r":    88.0,
                "hip_yaw_l":    88.0,
                "hip_roll_r":  139.0,
                "hip_roll_l":  139.0,
                "hip_pitch_r":  88.0,
                "hip_pitch_l":  88.0,
                "knee_pitch_r": 139.0,
                "knee_pitch_l": 139.0,
                "waist_yaw":    88.0,
            },
            # 各关节速度上限（单位：rad/s）
            # hip_roll / knee 速度较低，hip_yaw / waist 速度较高
            velocity_limit_sim={
                "hip_yaw_r":    32.0,
                "hip_yaw_l":    32.0,
                "hip_roll_r":   20.0,
                "hip_roll_l":   20.0,
                "hip_pitch_r":  32.0,
                "hip_pitch_l":  32.0,
                "knee_pitch_r": 20.0,
                "knee_pitch_l": 20.0,
                "waist_yaw":    32.0,
            },
            # PD 控制器的位置增益 Kp（刚度，单位：N·m/rad）
            # hip_pitch / knee / waist 刚度更高，提供更强的姿态支撑
            stiffness={
                "hip_yaw_r":    150.0,
                "hip_yaw_l":    150.0,
                "hip_roll_r":   150.0,
                "hip_roll_l":   150.0,
                "hip_pitch_r":  200.0,
                "hip_pitch_l":  200.0,
                "knee_pitch_r": 200.0,
                "knee_pitch_l": 200.0,
                "waist_yaw":    200.0,
            },
            # PD 控制器的速度增益 Kd（阻尼，单位：N·m·s/rad），抑制振荡
            damping={
                "hip_yaw_r":    5.0,
                "hip_yaw_l":    5.0,
                "hip_roll_r":   5.0,
                "hip_roll_l":   5.0,
                "hip_pitch_r":  5.0,
                "hip_pitch_l":  5.0,
                "knee_pitch_r": 5.0,
                "knee_pitch_l": 5.0,
                "waist_yaw":    5.0,
            },
            # 电机电枢惯量（单位：kg·m²），用于模拟电机转子惯量对动力学的影响
            armature=0.01,
        ),

        # ── 足部执行器（踝关节）────────────────────────────────
        # 踝关节力矩较小，刚度/阻尼参数更柔顺以适应地面接触
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_r", "ankle_pitch_l"],
            # 踝关节力矩上限（35 N·m，远小于髋/膝）
            effort_limit_sim={
                "ankle_pitch_r": 35.0,
                "ankle_pitch_l": 35.0,
            },
            # 踝关节速度上限（30 rad/s，摆动较快）
            velocity_limit_sim={
                "ankle_pitch_r": 30.0,
                "ankle_pitch_l": 30.0,
            },
            stiffness=20.0,   # 踝关节刚度（较腿部柔顺，利于接触稳定）
            damping=2.0,      # 踝关节阻尼
            armature=0.01,    # 电枢惯量
        ),

        # ── 手臂执行器（肩关节 + 肘关节）──────────────────────
        # 手臂主要用于行走时的配重摆臂，力矩需求较小
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_r",  # 右肩俯仰
                "shoulder_pitch_l",  # 左肩俯仰
                "elbow_pitch_r",     # 右肘俯仰
                "elbow_pitch_l",     # 左肘俯仰
            ],
            # 手臂关节力矩上限（25 N·m，最小）
            effort_limit_sim={
                "shoulder_pitch_r": 25.0,
                "shoulder_pitch_l": 25.0,
                "elbow_pitch_r":    25.0,
                "elbow_pitch_l":    25.0,
            },
            # 手臂关节速度上限（37 rad/s，摆臂速度较快）
            velocity_limit_sim={
                "shoulder_pitch_r": 37.0,
                "shoulder_pitch_l": 37.0,
                "elbow_pitch_r":    37.0,
                "elbow_pitch_l":    37.0,
            },
            stiffness=50.0,   # 手臂关节刚度（中等，保持姿态同时允许摆动）
            damping=2.0,      # 手臂关节阻尼
            armature=0.01,    # 电枢惯量
        ),
    },
)

from .base_config import BaseConfig

"""这个文件记录了详细的机器人配置：
环境，地形，域随机化，commend指令，PD控制，asset，奖励，噪声，归一化等等

"""

class LeggedRobotCfg(BaseConfig):
    class env:
        #强化学习环境中，同时训练智能体的数量，观测量。
        num_envs = 2048
        # 强化学习观测值的数量
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
		## 是否有特权观测值，所谓特权观测值就是实际机器人观测不到但仿真环境可以观测到的量，比如地面刚度，摩擦系数等
		## 如果该值不是None的话priviledge_obs_buf会通过step()函数返回，否则step()函数返回None

        # 可操控的动作数量
        num_actions = 12
        # 
        env_spacing = 3.  # not used with heightfields/trimeshes 
        # 向算法发送超时信息
        send_timeouts = True # send time out information to the algorithm
        # 机器人存活时间
        episode_length_s = 20 # episode length in seconds
        test = False

    ## 地形类，包含地形类型，大小，等初始化参数，用于terrain.py
    class terrain:
        ## 地形的表示形式：‘trimesh’是用三角网格，‘heightfield’是二维网格，有点类似与栅格的形式
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # 水平方向分辨率
        horizontal_scale = 0.1 # [m]
        # 垂直方向分辨率
        vertical_scale = 0.005 # [m]
        # 生成局部地形边界的大小
        border_size = 25 # [m]
        # 是否使用课程，课程的含义是，当机器人在当前环境下，运行情况较好后，增加地形的难度
        # curriculum = True
        curriculum = False
        
        # 静摩擦
        static_friction = 1.0
        # 滑动摩擦
        dynamic_friction = 1.0
        # 误差补偿
        restitution = 0.

        # 以下参数仅使用于粗糙地形
        # rough terrain only:
        measure_heights = True

        # 创建一个1.6*1的矩形
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 是否选择一个单一的独特地形
        # selected = False # select a unique terrain type and pass all arguments
        selected = True
        # 选择地形的字典值，即名称
        terrain_kwargs = None # Dict of arguments for selected terrain
        # terrain_kwargs = {'random':'random_uniform_terrain'} 
            # Dict of arguments for selected terrain
            # random_uniform_terrain(), sloped_terrain(), pyramid_sloped_terrain(), 
            # discrete_obstacles_terrain(), wave_terrain(), stairs_terrain(), 
            # pyramid_stairs_terrain(), and stepping_stones_terrain()

        # 初始化地形的状态等级
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        
        #生成的各种地形块，有几行
        num_rows= 10 # number of terrain rows (levels)
        # 生成的地形块，有几列
        num_cols = 20 # number of terrain cols (types)

        #地形类别包含：平坦坡，崎岖坡，正台阶，负台阶，离散地形。
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 各地形的比例
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only: 当坡度大于该阈值后直接修正成垂直墙
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        #terrain_generation_rule = 'curriculum'  #这是一个字符串，用于匹配，地形的生成类别，分别是“curriculum”，“fixed”，“random”，“obstest”（这是自定义的）

    # 指令类，机器人指令的设置参数
    class commands:
        # 是否使用课程
        curriculum = False
        # 课程难度的最高级
        max_curriculum = 1.
        # 指令的个数，默认是四个，x，y的线速度，角速度，航向。
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 指令更改的时间
        resampling_time = 10. # time before command are changed[s]
        # 是否为航向控制模式，航向模式下航向角速度基于航向偏差进行计算
        heading_command = True # if true: compute ang vel command from heading error
        # 指令范围
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # 机器人的初始状态
    class init_state:
        # 初始位置
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        # 初始姿态，利用四元数表示，目前设定的四元数计算的欧拉角都为0
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        #初始线速度，各方向都为0
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        # 初始角速度0
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # 初始关节位置都是0
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    # 机器人关节电机控制模式，以及参数
    class control:
        # 控制类型：位置控制，速度控制，扭矩控制
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        # PD驱动的参数
        # stiffness代表刚度系数k_p，damping代表阻尼系数k_d
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]

        # 公式如下,与action的转化为什么要有这样的比例因子还不清楚
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # 仿真环境的控制频率/decimation=实际环境中的控制频率
        decimation = 4

    class asset:
        # 存放urdf的位置，此处为空，之后具体的机器人.py继承此类然后赋值具体的urdf位置
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        # 接触惩罚
        penalize_contacts_on = []
        # 单个机器人终止的条件
        terminate_after_contacts_on = []
        # 取消重力
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">

        # 是否固定机器人本体
        fix_base_link = False # fixe the base of the robot
        # 默认的关节驱动模式
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # 是否开启自身碰撞检测，比如本体和腿部碰撞
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

        # 密度        
        density = 0.001
        # 角速度阻尼
        angular_damping = 0.
        linear_damping = 0.
        # 最大角速度
        max_angular_velocity = 1000.
        # 最大线速度
        max_linear_velocity = 1000.
        # 电驱
        armature = 0.
        thickness = 0.01

    # 随机化相关参数，增强训练出模型的鲁棒性
    class domain_rand:
        # 摩擦力是否有变化
        randomize_friction = True
        # 摩擦系数范围
        friction_range = [0.5, 1.25]
        # 本体质量是否有变化
        randomize_base_mass = False
        # 增加质量的分辨率
        added_mass_range = [-1., 1.]
        
        # 是否增加推动机器人的力,推动机器人力的时间间隔
        push_robots = True
        push_interval_s = 15
        # 最大推动速度
        max_push_vel_xy = 1.

    # 奖励函数类,定义了各个奖励函数
    class rewards:
        class scales:
            # 任务终止权重
            termination = -0.0
            # 跟踪线速度权重
            tracking_lin_vel = 1.0
            # 跟踪角速度权重
            tracking_ang_vel = 0.5
            # z轴线速度权重
            lin_vel_z = -2.0
            # 姿态方向角速度权重
            ang_vel_xy = -0.05
            # 机器人姿态权重
            orientation = -0.
            # 关节扭矩权重
            torques = -0.00001
            # 关节速度权重
            dof_vel = -0.
            # 关节加速度权重
            dof_acc = -2.5e-7
            # 本体高度权重
            base_height = -0. 
            # 足部悬空时间权重
            feet_air_time =  1.0
            # 本体碰撞权重
            collision = -1.
            # 避免垂直障碍接触权重
            feet_stumble = -0.0 
            # 动作变化率权重
            action_rate = -0.01
            # 无期望命令保持静止权重
            stand_still = -0.

        # 该项设置为true时,奖励为负的变为,这一项原因是避免过早终止的问题
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        #urdf限制区间,如果是则限制区间就按照urdf的机械限位决定,超出区间就惩罚
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            # 最多观测数
        clip_observations = 100.
        # 最多动作数，在step（）函数中被使用
        clip_actions = 100.

    # 增加噪声的分辨率
    class noise:
        add_noise = True
        # 在legged_robot.py的_get_noise_scale_vec()函数中用到
        #一个观测量（如线性速度）最终的噪声计算如式noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_level = 1.0 # scales other values
        class noise_scales:
            # 关节位置噪声分辨率
            dof_pos = 0.01
            # 关节速度噪声分辨率
            dof_vel = 1.5
            # 线性速度噪声分辨率
            lin_vel = 0.1
            # 角速度噪声分辨率
            ang_vel = 0.2
            # 重力噪声分辨率
            gravity = 0.05
            # 本体高度噪声分辨率
            height_measurements = 0.1

    # viewer camera:仿真的默认视角参数设置
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    # 仿真环境，仿真步长，物理属性
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    # 种子的设置，种子在make_env函数中被设置，用于控制随机化过程的初始化，确保实验的可重复性
    seed = 1
    # 运行类的名称
    runner_class_name = 'OnPolicyRunner'
    
    # 策略类参数
    class policy:
        init_noise_std = 1.0
        # 演员隐藏层维度
        actor_hidden_dims = [512, 256, 128]
        # 评论家隐藏层维度
        critic_hidden_dims = [512, 256, 128]
        # 激活函数类型
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid 
        # only for 'ActorCriticRecurrent': 接下来的参数只有ActorCriticRecurrent算法需要
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params训练的超参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        # 策略的类名
        policy_class_name = 'ActorCritic'
        # 算法的类名
        algorithm_class_name = 'PPO'
        # 每次迭代单个机器人的步长
        num_steps_per_env = 24 # per iteration
        # 策略训练次数
        max_iterations = 5000 # number of policy updates

        # logging
        # 存储训练模型的间隔，每50次存储一个模型
        save_interval = 50 # check for potential saves every this many iterations
        # 实验名称
        experiment_name = 'test'
        run_name = ''
        # load and resume
        # 是否接着上次训练
        resume = False
        # 读入上次训练情况
        load_run = -1 # -1 = last run
        # 读入上次训练存储的模型
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
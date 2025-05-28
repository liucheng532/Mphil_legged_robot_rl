import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
from math import sqrt

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments()

# 配置模拟参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.use_gpu = args.use_gpu

# 创建物理模拟器
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("Failed to create simulation")
    exit()

# # 加载球体资产（用于可视化）
# asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
# asset_file = "urdf/ball.urdf"
# asset_options = gymapi.AssetOptions()
# asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# # 设置环境参数
# num_envs = 800
# num_per_row = 80
# env_spacing = 0.56
# env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
# env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
# pose = gymapi.Transform()
# pose.r = gymapi.Quat(0, 0, 0, 1)
# pose.p.z = 1.
# pose.p.x = 3.

# envs = []
# np.random.seed(17)  # 设置随机种子

# for i in range(num_envs):
#     env = gym.create_env(sim, env_lower, env_upper, num_per_row)
#     envs.append(env)
#     c = 0.5 + 0.5 * np.random.random(3)
#     color = gymapi.Vec3(c[0], c[1], c[2])
#     ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
#     gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# # 保存初始状态
# initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# 地形参数设置
terrain_width = 12.0
terrain_length = 12.0
horizontal_scale = 0.25  # 水平精度 [m]
vertical_scale = 0.005   # 垂直精度 [m]
num_rows = int(terrain_width / horizontal_scale)
num_cols = int(terrain_length / horizontal_scale)
num_terains = 6  # 总共 6 种地形

# 创建高度场
heightfield = np.zeros((num_terains * num_rows, num_cols), dtype=np.int16)

# 定义子地形生成函数
def new_sub_terrain():
    return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

# 生成 6 种地形（确保边缘高度为 0）
# 1. 平坦地形
heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=0.0, max_height=0.0, step=0.1).height_field_raw

# 2. 金字塔斜坡
heightfield[num_rows:2*num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw

# 3. 障碍物地形
heightfield[2*num_rows:3*num_rows, :] = discrete_obstacles_terrain(
    new_sub_terrain(), max_height=0.5, min_size=1., max_size=5., num_rects=20
).height_field_raw

# 4. 波浪地形
heightfield[3*num_rows:4*num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2., amplitude=1.).height_field_raw

# 5. 金字塔阶梯
heightfield[4*num_rows:5*num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw

# 6. 平坦地形
heightfield[5*num_rows:6*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw

# 将高度场转换为三角形网格
vertices, triangles = convert_heightfield_to_trimesh(
    heightfield,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    slope_threshold=1.5
)

# 设置网格参数
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.

# 添加网格到仿真器
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

# 创建可视化窗口
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("Failed to create viewer")
    exit()

# 设置相机视角（覆盖 6 种地形）
cam_pos = gymapi.Vec3(-5, -5, 20)  # 提高相机高度以覆盖 72m 长度
cam_target = gymapi.Vec3(0, 0, 10)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 主循环
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


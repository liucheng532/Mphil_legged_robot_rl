import isaacgym
from isaacgym import gymapi, gymutil
from isaacgym import terrain_utils
import random
import numpy as np


# 初始化 Isaac Gym
gym = gymapi.acquire_gym()

# 设置模拟参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z  # 设置坐标系方向为 Z 轴向上
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # 重力加速度

# 创建物理模拟器
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("Failed to create simulation")
    exit()

# 创建一个环境
env = gym.create_env(sim, 
                     gymapi.Vec3(-10.0, -10.0, 0.0),  # 环境左下角坐标
                     gymapi.Vec3(10.0, 10.0, 10.0),   # 环境右上角坐标
                     1)

# 定义地形尺寸（4m × 4m）
terrain_size = 4.0
horizontal_scale = 0.1  # 水平精度（每个单元格 0.1m）
vertical_scale = 0.1    # 垂直精度
width = int(terrain_size / horizontal_scale)
length = int(terrain_size / horizontal_scale)

# 生成波浪地形
terrain = terrain_utils.SubTerrain(
    width=width,
    length=length,
    vertical_scale=vertical_scale,
    horizontal_scale=horizontal_scale
)
terrain_utils.wave_terrain(terrain, num_waves=3, amplitude=0.1)

# 将地形转换为三角形网格
vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
    terrain.height_field_raw,
    horizontal_scale=terrain.horizontal_scale,
    vertical_scale=terrain.vertical_scale
)

# 创建三角形网格参数
mesh_params = gymapi.TriangleMeshParams()
mesh_params.nb_vertices = len(vertices)
mesh_params.nb_triangles = len(triangles)
mesh_params.transform.p = gymapi.Vec3(0.0, 0.0, 0.0)  # 设置地形位置
mesh_params.static_friction = 0.5
mesh_params.dynamic_friction = 0.5
mesh_params.restitution = 0.0

print("Vertices shape:", vertices.shape)      # 应为 (1600, 3)
print("Triangles shape:", triangles.shape)    # 应为 (2882, 3)
# 添加地形到环境
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), mesh_params)

# 创建可视化窗口
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("Could not create viewer")
    exit()

# 设置相机视角（俯视地形）
cam_pos = gymapi.Vec3(0.0, 0.0, 10.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 主循环：保持窗口打开
while not gym.query_viewer_has_closed(viewer):
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
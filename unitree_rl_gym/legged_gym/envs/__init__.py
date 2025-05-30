#机器人，环境注册脚本

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR



from legged_gym.envs.g1.g1_random_terrain_config import G1RandomTerrainRoughCfg, G1RandomTerrainRoughCfgPPO
from legged_gym.envs.g1.g1_mix_terrain import G1MixTerrainRoughCfg, G1MixTerrainRoughCfgPPO

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

task_registry.register( "g1_random_terrain", G1Robot, G1RandomTerrainRoughCfg(), G1RandomTerrainRoughCfgPPO())

task_registry.register( "g1_mix_terrain", G1Robot, G1MixTerrainRoughCfg(), G1MixTerrainRoughCfgPPO())

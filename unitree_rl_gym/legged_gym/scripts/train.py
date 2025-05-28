import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *  # 执行envs文件夹里的 init.py 
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    ##接收终端启动程序时给的参数args，返回环境env以及环境初始参数“env_cfg”
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    ##进行相关算法配置  
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
##该框架的入口
if __name__ == '__main__':
    ##首先执行获取参数的函数，该函数定义在utils/helper.py中
    args = get_args()
    train(args)




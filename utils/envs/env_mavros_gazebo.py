#!/usr/bin/env python3


import numpy as np
import os

from .env_base import EnvBase
from .mavros_related.mavros_env import MavrosEnv

"""
    算法与MAVROS环境的接口
    与mavros中相关的msg和srv文件的用法，参考
    https://wiki.ros.org/mavros
    https://wiki.ros.org/mavros_extras
    这两个官方网址里面提供的解析（写得很难让人读懂）
    可以部分参考
    https://blog.csdn.net/z1872385/article/details/124606883
    或者参考本人整理的rospy_related.xmind文件
"""


class EnvMavrosGazebo(EnvBase):
    def __init__(
        self,
        seed_range=150,
        privileged=False,
    ):
        self.env = MavrosEnv()
        self.seed_range = seed_range
        self.privileged = privileged

        # settings.sh中有两条命令行指令，分别是打开roscore和载入launch文件
        # os.system("sh mavros_related/settings/settings.sh")

    def step(self, action):
        obs, reward, done = self.env.step(action=action)
        if not self.privileged:
            return reward, obs, done
        else:
            return reward, obs, obs, done

    def reset(self):
        seed = np.random.randint(low=0, high=self.seed_range)
        state_array = self.env.reset(seed=seed)
        if not self.privileged:
            return state_array
        else:
            return state_array, state_array

    def get_dims_dict(self):
        dims_dict = {}
        dims_dict["actor_state_dim"] = self.env.observation_dim
        dims_dict["critic_state_dim"] = self.env.observation_dim
        dims_dict["action_dim"] = self.env.action_dim
        return dims_dict

    def get_range(self):
        action_range = self.env.action_range
        if not self.privileged:
            return None, None, action_range[0], action_range[1]
        else:
            return (
                None,
                None,
                None,
                None,
                action_range[0],
                action_range[1],
            )

import gymnasium as gym
import numpy as np

from .env_base import EnvBase

"""
    算法与环境的接口，在将算法送给无人机虚拟环境前，先在小环境上测试一下
    利用OpenAI Gymnasium测试所写的程序，版本号，1.0.0
"""


class EnvGymMCC(EnvBase):
    """
    这里采用MountainCarContinuous-v0作为测试环境，该环境的详细描述可参考，
    https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
    """

    def __init__(
        self,
        name="MountainCarContinuous-v0",
        seed_range=150,
        privileged=False,
        render_mode="rgb_array",
    ):
        self.env = gym.make(id=name, render_mode=render_mode, goal_velocity=0.1)
        self.seed_range = seed_range
        self.privileged = privileged

    def step(self, action):
        obs, reward, done1, done2, _ = self.env.step(action=action)
        done = done1 or done2
        reward = reward * 1000
        if not self.privileged:
            return reward, obs, done
        else:
            return reward, obs, obs, done

    def reset(self):
        seed = np.random.randint(low=0, high=self.seed_range)
        state_array, _ = self.env.reset(seed=seed)
        if not self.privileged:
            return state_array
        else:
            return state_array, state_array

    def get_dims_dict(self):
        obs_box = self.env.observation_space  # Gymnasium Box类
        action_box = self.env.action_space
        dims_dict = {}
        dims_dict["actor_state_dim"] = obs_box.shape[0]
        dims_dict["critic_state_dim"] = obs_box.shape[0]
        dims_dict["action_dim"] = action_box.shape[0]
        return dims_dict

    def get_action_range(self):
        obs_box = self.env.observation_space  # Gymnasium Box类
        action_box = self.env.action_space
        if not self.privileged:
            return obs_box.low, obs_box.high, action_box.low, action_box.high
        else:
            return (
                obs_box.low,
                obs_box.high,
                obs_box.low,
                obs_box.high,
                action_box.low,
                action_box.high,
            )

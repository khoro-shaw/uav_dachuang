#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time

import os
import sys
import rospy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.envs import EnvGymMCC, EnvGymPen, EnvMavrosGazebo
from utils.algorithms import DDPGBase, PPOBase


class FCNet(nn.Module):
    def __init__(self, actor_param_list, state_dim, action_dim):
        super(FCNet, self).__init__()
        self.fc_list = []
        self.act_list = []
        self.num = len(actor_param_list)
        for idx, num in enumerate(actor_param_list):
            if idx == 0:
                self.fc0 = nn.Linear(
                    in_features=state_dim,
                    out_features=actor_param_list[idx],
                )
                self.act0 = self.get_activation()
                self.fc_list.append(self.fc0)
                self.act_list.append(self.act0)
            else:
                exec(
                    f"self.fc{idx} = nn.Linear(in_features={actor_param_list[idx - 1]}, out_features={num})"
                )
                exec(f"self.act{idx} = self.get_activation()")
                exec(f"self.fc_list.append(self.fc{idx})")
                exec(f"self.act_list.append(self.act{idx})")

            self.fc_mu = nn.Linear(
                in_features=actor_param_list[-1], out_features=action_dim
            )
            self.act_mu = self.get_activation("tanh")
            self.fc_sigma = nn.Linear(
                in_features=actor_param_list[-1], out_features=action_dim
            )
            self.act_sigma = self.get_activation("softplus")

    def forward(self, x):
        for i in range(self.num):
            x = self.act_list[i](self.fc_list[i](x))
        mu = 2.0 * self.act_mu(self.fc_mu(x))
        std = self.act_sigma(self.fc_sigma(x))
        return mu, std

    def get_activation(self, act="relu"):
        if act == "elu":
            return nn.ELU()
        elif act == "selu":
            return nn.SELU()
        elif act == "relu":
            return nn.ReLU()
        elif act == "crelu":
            return nn.CReLU()
        elif act == "lrelu":
            return nn.LeakyReLU()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "softplus":
            return nn.Softplus()
        else:
            print("invalid activation function")
            return None


class BehaviorClone:
    def __init__(
        self,
        actor_param_list=[128, 64, 16],
    ):
        self.policy = FCNet(actor_param_list, state_dim=10, action_dim=6)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(), lr=1e-3)

    def learn(self, state_array, action_array):
        state_tensor = torch.tensor(data=state_array, dtype=torch.float)
        action_tensor = torch.tensor(data=action_array, dtype=torch.float)
        mu_tensor, std_tensor = self.policy(state_tensor)
        action_dist = torch.distributions.Normal(loc=mu_tensor, scale=std_tensor)
        log_prob_tensor = action_dist.log_prob(action_tensor)
        loss = -torch.mean(log_prob_tensor)  # 最大似然估计

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def take_one_step(self, state_array):
        state_tensor = torch.tensor(state_array)
        mu_tensor, std_tensor = self.policy(state_tensor)
        action_dist = torch.distributions.Normal(loc=mu_tensor, scale=std_tensor)
        action_tensor = action_dist.sample()
        return action_tensor


# -----------------------------------------------------
# 从csv文件中取数据
csv_files = os.listdir("./expert_data/")
expert_state_list = []
expert_action_list = []
for file in csv_files:
    df = pd.read_csv("./expert_data/" + file)
    state_array = df[  # dim = 10
        [
            "x_diff",
            "y_diff",
            "z_diff",
            "yaw_diff",
            "x_current",
            "y_current",
            "z_current",
            "roll_current",
            "pitch_current",
            "yaw_current",
        ]
    ].to_numpy()
    action_array = df[  # dim = 6
        [
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel_x",
            "ang_vel_y",
            "ang_vel_z",
        ]
    ].to_numpy()
    expert_state_list.append(state_array)
    expert_action_list.append(action_array)

expert_state_array = np.array(expert_state_list).reshape((-1, 10))
expert_action_array = np.array(expert_action_list).reshape((-1, 6))


imitator_bc = BehaviorClone()
episode_num = 5000
batch_size = 128
for i in range(episode_num):
    random_indices = torch.randint(
        low=0, high=len(expert_action_array), size=(batch_size,)
    )
    imitator_bc.learn(
        state_array=expert_state_array[random_indices],
        action_array=expert_action_array[random_indices],
    )

t = time.localtime()
folder = os.path.exists("./expert_data/bc/latest")
if not folder:
    os.makedirs("./expert_data/bc/latest")

torch.save(
    imitator_bc.policy.state_dict(),
    "./expert_data/bc/latest/cloned.pth",
)
torch.save(
    imitator_bc.policy.state_dict(),
    f"./expert_data/bc/cloned{t.tm_mday}_{t.tm_hour}_{t.tm_min}.pth",
)

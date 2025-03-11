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


class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Discriminator, self).__init__()
        # 观测值维度+动作维度
        self.fc1 = nn.Linear(in_features=obs_dim + action_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x, a):
        cat = torch.cat(tensors=[x, a], dim=-1)
        x = F.relu(self.fc1(cat))
        return F.sigmoid(self.fc2(x))  # output: 1 -> agent; 0 -> expert


class GAIL:
    def __init__(
        self,
        actor_param_list=[128, 64, 16],
        critic_param_list=[128, 32, 8],
        params_dict={
            "tuple_num": 5000,
            "batch_size": 512,
            "gamma": 0.5,
            "epochs": 200,
            "eps": 0.2,
            "critic_lr": 1e-2,
            "critic_eps": 8e-2,
            "actor_lr": 1e-2,
            "actor_eps": 8e-2,
            "sigma": 1e-2,
            "tau": 5e-3,
            "lmbda": 0.95,
        },
        load_flag=True,
    ):
        self.discriminator = Discriminator(obs_dim=10, action_dim=5)
        self.optim_discrim = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=1e-3
        )

        self.agent = PPOBase(
            env=self.env,
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            params_dict=params_dict,
            load_flag=load_flag,
        )
        self.env = EnvMavrosGazebo()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def learn(
        self,
        expert_state_array,
        expert_action_array,
        agent_state,
        agent_action,
        next_state_array,
        done_list,
    ):
        expert_state_tensor = torch.tensor(
            data=expert_state_array, dtype=torch.float
        ).to(self.device)
        expert_action_tensor = torch.tensor(
            data=expert_action_array, dtype=torch.float
        ).to(self.device)
        agent_state_tensor = agent_state.to(
            self.device
        )  # PPO训练时已将env给的array转换成Tensor
        agent_action_tensor = agent_action.to(
            self.device
        )  # PPO训练时已将env给的array转换成Tensor

        expert_prob_tensor = self.discriminator(
            expert_state_tensor, expert_action_tensor
        )
        agent_prob_tensor = self.discriminator(agent_state_tensor, agent_action_tensor)

        # discriminator是把expert判断成0，agent判断成1
        loss = nn.BCELoss()(
            agent_prob_tensor, torch.ones_like(agent_prob_tensor)
        ) + nn.BCELoss()(expert_prob_tensor, torch.zeros_like(expert_prob_tensor))

        self.optim_discrim.zero_grad()
        loss.backward()
        self.optim_discrim.step()

        reward_list = -torch.log(agent_prob_tensor).detach().cpu().tolist()

        self.agent.tuples_log = zip(
            agent_state_tensor,
            agent_action_tensor,
            reward_list,
            next_state_array,
            done_list,
        )
        self.agent.update()


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

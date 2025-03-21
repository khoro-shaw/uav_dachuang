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
    ):
        self.discriminator = Discriminator(obs_dim=10, action_dim=6)
        self.optim_discrim = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=1e-3
        )

        self.env = EnvMavrosGazebo()
        self.agent = PPOBase(
            env=self.env,
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            params_dict=params_dict,
            load_flag=True,
            imitation_flag=True,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def learn(
        self,
        expert_state_array,
        expert_action_array,
        agent_state_array,
        agent_action_array,
        next_state_list,
        done_list,
    ):
        expert_state_tensor = torch.tensor(
            data=expert_state_array, dtype=torch.float
        ).to(self.device)
        expert_action_tensor = torch.tensor(
            data=expert_action_array, dtype=torch.float
        ).to(self.device)

        agent_state_tensor = torch.tensor(data=agent_state_array, dtype=torch.float).to(
            self.device
        )
        agent_action_tensor = torch.tensor(
            data=agent_action_array, dtype=torch.float
        ).to(self.device)

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

        reward_array = -torch.log(agent_prob_tensor).detach().cpu().numpy()
        next_state_array = np.array(next_state_list)
        done_array = np.array(done_list)

        # rospy.loginfo(agent_state_tensor)
        # rospy.loginfo(agent_state_tensor.shape)
        # rospy.loginfo("-" * 15)
        # rospy.loginfo(agent_action_tensor)
        # rospy.loginfo(agent_action_tensor.shape)
        # rospy.loginfo("-" * 20)
        # rospy.loginfo(reward_tensor)
        # rospy.loginfo(reward_tensor.shape)
        # rospy.loginfo("-" * 25)
        # rospy.loginfo(next_state_tensor)
        # rospy.loginfo(next_state_tensor.shape)
        # rospy.loginfo("-" * 30)
        # rospy.loginfo(done_tensor)
        # rospy.loginfo(done_tensor.shape)
        # rospy.loginfo("-" * 35)

        self.agent.tuples_log_imitation = zip(
            agent_state_array,
            agent_action_array,
            reward_array,
            next_state_array,
            done_array,
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


imitator_gail = GAIL()
episode_num = 500
for i in range(episode_num):
    done = False
    agent_state = imitator_gail.agent.env.reset()
    agent_state_list = []
    agent_action_list = []
    next_state_list = []
    done_list = []
    i = 0
    while not done:
        i += 1
        agent_state_tensor = torch.tensor(data=agent_state, dtype=torch.float)
        agent_action_tensor, next_state, done, _ = imitator_gail.agent.take_one_step(
            agent_state_tensor
        )
        agent_state_list.append(agent_state)  # ndarray
        agent_action_list.append(agent_action_tensor.numpy())  # Tensor
        next_state_list.append(next_state)  # ndarray
        done_list.append(done)  # bool

        agent_state = next_state

    agent_state_arrays = np.array(agent_state_list)
    agent_action_arrays = np.array(agent_action_list)

    imitator_gail.learn(
        expert_state_array,
        expert_action_array,
        agent_state_arrays,
        agent_action_arrays,
        next_state_list,
        done_list,
    )

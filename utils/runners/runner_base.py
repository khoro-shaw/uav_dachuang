import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from envs import EnvGymMCC
from algorithms import DDPGBase


class RunnerBase:
    """
    训练算法的类
    """

    def __init__(
        self,
        actor_param_list,
        critic_param_list,
        params_dict,
        track_num=10,
        update_num=1000,
        env_class="EnvGymMCC",
        alg_class="DDPGBase",
    ):
        self.env = eval(env_class)()
        self.env_test = eval(env_class)(render_mode="human")
        self.alg = eval(alg_class)(
            env=self.env,
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            params_dict=params_dict,
        )
        self.alg_test = eval(alg_class)(
            env=self.env_test,
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            params_dict=params_dict,
        )
        self.track_num = track_num
        self.update_num = update_num
        self.actor_loss_log = []
        self.critic_loss_log = []
        self.reward_log = []

    def train(self, load=False):
        a_loss_log = []
        c_loss_log = []
        r_log_log_log = []
        if load:
            self.alg.load_model()
        for i in range(self.update_num):
            r_log_log = []
            for j in range(self.track_num):
                r_log = self.alg.take_one_track()  # take one track
                r_log_log.append(np.mean(r_log).item())
            a_loss, c_loss = self.alg.update()  # update
            a_loss_log.append(a_loss)
            c_loss_log.append(c_loss)
            r_log_log_log.append(np.mean(r_log_log).item())
            if (i + 1) % 10 == 0:
                self.actor_loss_log.append(torch.mean(torch.tensor(a_loss_log)).item())
                self.critic_loss_log.append(torch.mean(torch.tensor(c_loss_log)).item())
                self.reward_log.append(np.mean(r_log_log_log).item())
                print(
                    f"update {i}, actor_loss: {self.actor_loss_log[-1]}, critic_loss: {self.critic_loss_log[-1]}, average reward: {self.reward_log[-1]}",
                )
                a_loss_log = []
                c_loss_log = []
                r_log_log_log = []
        self.alg.save_model()

    def visualize(self):
        update_log = list(range(len(self.actor_loss_log)))
        plt.plot(update_log, self.actor_loss_log)
        plt.xlabel("updates")
        plt.ylabel("actor_loss")
        plt.title(f"actor_loss per 10 updates")
        plt.show()

        plt.plot(update_log, self.critic_loss_log)
        plt.xlabel("updates")
        plt.ylabel("critic_loss")
        plt.title(f"critic_loss per 10 updates")
        plt.show()

        plt.plot(update_log, self.reward_log)
        plt.xlabel("updates")
        plt.ylabel("average reward")
        plt.title(f"average reward per 10 updates")
        plt.show()

        self.alg_test.load_model()
        self.alg_test.take_one_track()


actor_param_list = [16, 32, 8]
critic_param_list = [16, 32, 8]
params_dict = {
    "tuple_num": 100000,
    "batch_size": 1024,
    "gamma": 0.5,
    "critic_lr": 1e-3,
    "critic_eps": 8e-2,
    "actor_lr": 1e-3,
    "actor_eps": 8e-2,
}

runner = RunnerBase(
    actor_param_list=actor_param_list,
    critic_param_list=critic_param_list,
    params_dict=params_dict,
)

runner.train()
runner.visualize()

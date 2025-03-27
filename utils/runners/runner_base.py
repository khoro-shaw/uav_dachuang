import torch
import matplotlib.pyplot as plt

import os
import sys

# import rospy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from envs import EnvGymMCC, EnvGymPen, EnvGymWalker

# from envs import EnvMavrosGazebo


from algorithms import DDPGBase, PPOBase


class RunnerBase:
    """
    训练算法的类
    """

    def __init__(
        self,
        actor_param_list,
        critic_param_list,
        params_dict,
        track_num=5,
        update_num=500,
        env_class="EnvGymMCC",
        alg_class="PPOBase",
        load_flag=False,
    ):
        self.env = eval(env_class)()
        self.env_class = env_class
        self.env_test = (
            eval(env_class)(render_mode="human")
            if env_class != "EnvMavrosGazebo"
            else eval(env_class)()
        )
        self.alg = eval(alg_class)(
            env=self.env,
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            params_dict=params_dict,
            load_flag=load_flag,
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

    def train(self, load=False):
        a_loss_log = []
        c_loss_log = []
        if load:
            self.alg.load_model()
        for i in range(self.update_num):
            if self.env_class == "DDPGBase":  # 离线策略算法
                for j in range(self.track_num):
                    self.alg.take_one_track()  # take one track
            else:
                self.alg.take_one_track()
            a_loss, c_loss = self.alg.update(update_id=i)  # update

            a_loss_log.append(a_loss)
            c_loss_log.append(c_loss)
            if (i + 1) % 10 == 0:
                self.actor_loss_log.append(
                    torch.mean(torch.tensor(a_loss_log[-10:])).item()
                )
                self.critic_loss_log.append(
                    torch.mean(torch.tensor(c_loss_log[-10:])).item()
                )
                if self.env_class != "EnvMavrosGazebo":
                    print(
                        f"update {i}, actor_loss: {self.actor_loss_log[-1]}, critic_loss: {self.critic_loss_log[-1]}",
                    )
                else:
                    rospy.loginfo(
                        f"update {i}, actor_loss: {self.actor_loss_log[-1]}, critic_loss: {self.critic_loss_log[-1]}"
                    )
                a_loss_log = []
                c_loss_log = []
        self.alg.save_model()

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

    def visualize(self):
        self.alg_test.load_model()
        self.alg_test.take_one_track()

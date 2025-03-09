import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import rospy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from modules import ActorCriticProbs


class PPOBase:
    """
    PPO算法
    经典的在线策略算法
    基本思路：
    （1）与环境交互得到一条路径
    （2）actor网络给出各个维度的action值，比如前进的线速度大小，旋转的角速度大小之类的
    （3）critic网络给出Q_{t}值
    （4）reward + γ*Q_{t+1}给出critic的目标值
    （5）actor网络的目标，是使得，替代目标，surroga_obj，的值最大化
    参数由env的dims_dict，和独立的params_dict给出
    params_dict["gamma"]
    params_dict["sigma"]
    params_dict["lmbda"]
    params_dict["epochs"]
    params_dict["eps"]
    params_dict["critic_lr"]
    params_dict["critic_eps"]
    params_dict["actor_lr"]
    params_dict["actor_eps"]
    """

    def __init__(
        self,
        env,
        actor_param_list=None,
        critic_param_list=None,
        params_dict=None,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    ):
        self.env = env
        self.actor_critic = ActorCriticProbs(
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            env=self.env,
        )
        self.device = device
        self.gamma = params_dict["gamma"]
        self.critic_lr = params_dict["critic_lr"]
        self.critic_eps = params_dict["critic_eps"]
        self.actor_lr = params_dict["actor_lr"]
        self.actor_eps = params_dict["actor_eps"]
        self.sigma = params_dict["sigma"]
        self.lmbda = params_dict["lmbda"]
        self.epochs = params_dict["epochs"]
        self.eps = params_dict["eps"]
        self.dims_dict = self.env.get_dims_dict()
        self.tuples_log = []
        self.action_dist = None
        self.obs_low, self.obs_high, self.act_low, self.act_high = self.env.get_range()
        self.critic_optimizer = torch.optim.Adam(
            params=self.actor_critic.critic.parameters(),
            lr=self.critic_lr,
            eps=self.critic_eps,
        )
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor_critic.actor.parameters(),
            lr=self.actor_lr,
            eps=self.actor_eps,
        )

    def take_one_step(self, actor_state, critic_state=None):

        mu_tensor, sigma_tensor = self.actor_critic.actor(actor_state)
        # rospy.loginfo(mu_tensor)
        # rospy.loginfo(sigma_tensor)
        # rospy.loginfo("-" * 15)

        self.action_dist = torch.distributions.Normal(loc=mu_tensor, scale=sigma_tensor)

        action_tensor = self.action_dist.sample()
        # rospy.loginfo(action_tensor)

        action = (
            self.act_high - self.act_low
        ) * action_tensor.detach().numpy() + self.act_low

        # 加噪声，提升explore
        action += self.sigma * (
            (self.act_high - self.act_low)
            * np.random.random(self.dims_dict["action_dim"])
        )
        # rospy.loginfo(action)

        action = torch.where(action > self.act_high, self.act_high, action)
        action = torch.where(action < self.act_low, self.act_low, action)
        # rospy.loginfo(action)

        # rospy.loginfo(action)
        # rospy.loginfo("-" * 15)

        if critic_state is None:
            reward, next_state, done = self.env.step(action=action)
            self.tuples_log.append((actor_state, action, reward, next_state, done))
            return next_state, done, reward
        else:
            reward, next_actor_state, next_critic_state, done = self.env.step(
                action=action
            )
            self.tuples_log.append(
                (
                    actor_state,
                    critic_state,
                    action,
                    reward,
                    next_actor_state,
                    next_critic_state,
                    done,
                )
            )
            return next_actor_state, next_critic_state, done, reward

    def take_one_track(self):
        privileged = self.env.privileged
        done = False
        state = self.env.reset()
        state_tensor = torch.tensor(data=state, dtype=torch.float)

        self.tuples_log = []  # tuples_log更新
        if not privileged:
            while not done:
                state, done, reward = self.take_one_step(actor_state=state_tensor)
                state_tensor = torch.tensor(data=state, dtype=torch.float)

        else:
            actor_state_tensor = state_tensor
            critic_state_tensor = state_tensor
            while not done:
                actor_state, critic_state, done, reward = self.take_one_step(
                    actor_state=actor_state_tensor, critic_state=critic_state_tensor
                )
                actor_state_tensor = torch.tensor(data=actor_state, dtype=torch.float)
                critic_state_tensor = torch.tensor(data=critic_state, dtype=torch.float)

    def update(self):
        privileged = self.env.privileged
        if not privileged:
            # 在线策略，一条路径对应一次更新
            self.take_one_track()

            state, action, reward, next_state, done = zip(*self.tuples_log)
            # Tensor, Tensor, float64, np.array, bool

            # state = np.array(state)
            # action = np.array(action)
            reward = np.array(reward)
            next_state = np.array(next_state)
            done = np.array(done)

            state_tensor = torch.stack(tensors=state).to(device=self.device)
            action_tensor = torch.stack(tensors=action).to(device=self.device)
            reward_tensor = (
                torch.tensor(data=reward, dtype=torch.float)
                .view(-1, 1)
                .to(device=self.device)
            )
            next_state_tensor = torch.tensor(data=next_state, dtype=torch.float).to(
                device=self.device
            )
            done_tensor = (
                torch.tensor(data=done, dtype=torch.float)
                .view(-1, 1)
                .to(device=self.device)
            )

            td_target = reward_tensor + self.gamma * self.actor_critic.critic(
                next_state_tensor
            ) * (1 - done_tensor)
            td_delta = td_target - self.actor_critic.critic(state_tensor)

            # GAE算法估计优势，Advantage = Q - Value
            advantage = torch.zeros_like(td_delta[0])
            advantage_log = []
            for data in reversed(td_delta):
                advantage = (data + advantage * self.lmbda * self.gamma).item()
                advantage_log.append(advantage)

            advantage_tensor = torch.tensor(
                data=advantage_log[::-1], dtype=torch.float
            ).unsqueeze(dim=-1)

            old_log_prob = self.action_dist.log_prob(value=action_tensor)

            for i in range(self.epochs):
                mu, sigma = self.actor_critic.actor(state_tensor)
                new_action_dist = torch.distributions.Normal(loc=mu, scale=sigma)
                new_log_prob = new_action_dist.log_prob(action_tensor)

                ratio = torch.exp(new_log_prob - old_log_prob)

                surrogate_obj_1 = ratio * advantage_tensor
                surrogate_obj_2 = (
                    torch.clamp(input=ratio, min=1 - self.eps, max=1 + self.eps)
                    * advantage_tensor
                )

                actor_loss = torch.mean(-torch.min(surrogate_obj_1, surrogate_obj_2))
                critic_loss = torch.mean(
                    F.mse_loss(
                        self.actor_critic.critic(state_tensor), td_target.detach()
                    )
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            return actor_loss, critic_loss
        else:
            pass

    def save_model(self):
        t = time.localtime()

        folder = os.path.exists(
            f"./logs/ppo/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}"
        )
        if not folder:
            os.makedirs(
                f"./logs/ppo/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}"
            )

        folder = os.path.exists("./logs/ppo/latest")
        if not folder:
            os.makedirs("./logs/ppo/latest")

        torch.save(
            self.actor_critic.actor.state_dict(),
            f"./logs/ppo/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}/actor.pth",
        )
        torch.save(
            self.actor_critic.critic.state_dict(),
            f"./logs/ppo/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}/critic.pth",
        )
        torch.save(
            self.actor_critic.actor.state_dict(),
            "./logs/ppo/latest/actor.pth",
        )
        torch.save(
            self.actor_critic.critic.state_dict(),
            "./logs/ppo/latest/critic.pth",
        )

    def load_model(self, zeit=None):
        if zeit is None:
            state_dict_actor = torch.load("./logs/ppo/latest/actor.pth")
            state_dict_critic = torch.load("./logs/ppo/latest/critic.pth")
        else:
            state_dict_actor = torch.load("./logs/ppo/" + zeit + "/actor.pth")
            state_dict_critic = torch.load("./logs/ppo/" + zeit + "/critic.pth")
        self.actor_critic.actor.load_state_dict(state_dict_actor)
        self.actor_critic.critic.load_state_dict(state_dict_critic)

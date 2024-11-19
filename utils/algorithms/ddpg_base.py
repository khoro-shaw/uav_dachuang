import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from modules import ActorCriticBase
from storages import Storage


class DDPGBase:
    """
    DDPG算法，非常简单实现
    离线策略算法，收敛速度，以及效果，均不如在线算法
    基本思路：
    （1）与环境交互得到一条路径，把每条路径分块成一个个元组，每个元组作为一个数据单元
    （2）actor网络给出各个维度的action值，比如前进的线速度大小，旋转的角速度大小之类的
    （3）critic网络给出Q_{t}值
    （4）reward + γ*Q_{t+1}给出critic的目标值
    （5）actor网络的目标，是使得Q{t}值最大化
    参数由env的dims_dict，和独立的params_dict给出
    params_dict["tuple_num"]
    params_dict["batch_size"]
    params_dict["gamma"]
    params_dict["critic_lr"]
    params_dict["critic_eps"]
    params_dict["actor_lr"]
    params_dict["actor_eps"]
    params_dict["sigma"]
    params_dict["tau"]
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
        self.actor_critic = ActorCriticBase(
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            env=self.env,
        )
        self.actor_critic_target = ActorCriticBase(
            actor_param_list=actor_param_list,
            critic_param_list=critic_param_list,
            env=self.env,
        )
        self.actor_critic_target.actor.load_state_dict(
            self.actor_critic.actor.state_dict()
        )
        self.actor_critic_target.critic.load_state_dict(
            self.actor_critic.critic.state_dict()
        )
        self.device = device
        self.rl_tuples_log = Storage(total_length=params_dict["tuple_num"])
        self.gamma = params_dict["gamma"]
        self.batch_size = params_dict["batch_size"]
        self.critic_lr = params_dict["critic_lr"]
        self.critic_eps = params_dict["critic_eps"]
        self.actor_lr = params_dict["actor_lr"]
        self.actor_eps = params_dict["actor_eps"]
        self.sigma = params_dict["sigma"]
        self.tau = params_dict["tau"]
        self.dims_dict = self.env.get_dims_dict()
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

        action_tensor = self.actor_critic.actor(actor_state)
        action = (self.act_high - self.act_low) * action_tensor.detach().numpy()

        # 加噪声，提升explore
        action += self.sigma * (
            (self.act_high - self.act_low)
            * np.random.random(self.dims_dict["action_dim"])
        )

        if action > self.act_high:
            action = self.act_high
        if action < self.act_low:
            action = self.act_low

        if critic_state is None:
            reward, next_state, done = self.env.step(action=action)
            self.rl_tuples_log.append((actor_state, action, reward, next_state, done))
            return next_state, done, reward
        else:
            reward, next_actor_state, next_critic_state, done = self.env.step(
                action=action
            )
            self.rl_tuples_log.append(
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
            state, action, reward, next_state, done = self.rl_tuples_log.sample(
                batch_size=self.batch_size
            )
            state_tensor = torch.tensor(data=state, dtype=torch.float).to(
                device=self.device
            )
            action_tensor = torch.tensor(data=action, dtype=torch.float).to(
                device=self.device
            )
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

            next_action_tensor = self.actor_critic_target.actor(next_state_tensor)

            next_sa_tensor = torch.cat((next_state_tensor, next_action_tensor), dim=1)

            q_target = reward_tensor + self.gamma * (
                self.actor_critic_target.critic(next_sa_tensor)
            ) * (1 - done_tensor)

            sa_tensor = torch.cat((state_tensor, action_tensor), dim=1)

            q_calculated = self.actor_critic.critic(sa_tensor)

            # print(f"q_cal: {q_calculated.view(1,-1)}")
            # print(f"q_tar: {q_target.view(1,-1)}")

            critic_loss = F.mse_loss(input=q_calculated, target=q_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -torch.mean(
                self.actor_critic.critic(
                    torch.cat(
                        (state_tensor, self.actor_critic.actor(state_tensor)), dim=1
                    )
                )
            )

            # print(
            #     f"act_loss: {self.actor_critic.critic(torch.cat((state_tensor, self.actor_critic.actor(state_tensor)), dim=1)).view(1,-1)}"
            # )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft_update
            for param_target, param in zip(
                self.actor_critic_target.actor.parameters(),
                self.actor_critic.actor.parameters(),
            ):
                param_target.data.copy_(
                    param_target.data * (1.0 - self.tau) + param.data * self.tau
                )
            for param_target, param in zip(
                self.actor_critic_target.critic.parameters(),
                self.actor_critic.critic.parameters(),
            ):
                param_target.data.copy_(
                    param_target.data * (1.0 - self.tau) + param.data * self.tau
                )

            return actor_loss, critic_loss
        else:
            (
                actor_state,
                critic_state,
                action,
                reward,
                next_actor_state,
                next_critic_state,
                done,
            ) = self.rl_tuples_log.sample(batch_size=self.batch_size)
            actor_state_tensor = torch.tensor(data=actor_state, dtype=torch.float).to(
                device=self.device
            )
            critic_state_tensor = torch.tensor(data=critic_state, dtype=torch.float).to(
                device=self.device
            )
            action_tensor = torch.tensor(data=action, dtype=torch.float).to(
                device=self.device
            )
            reward_tensor = (
                torch.tensor(data=reward, dtype=torch.float)
                .view(-1, 1)
                .to(device=self.device)
            )
            next_actor_state_tensor = torch.tensor(
                data=next_actor_state, dtype=torch.float
            ).to(device=self.device)
            next_critic_state_tensor = torch.tensor(
                data=next_critic_state, dtype=torch.float
            ).to(device=self.device)
            done_tensor = (
                torch.tensor(data=done, dtype=torch.float)
                .view(-1, 1)
                .to(device=self.device)
            )
            next_action_tensor = self.actor_critic.actor(next_critic_state_tensor)
            next_critic_sa_tensor = torch.cat(
                (next_critic_state_tensor, next_action_tensor), dim=1
            )
            q_target = reward_tensor + self.gamma * self.actor_critic.critic(
                next_critic_sa_tensor
            ) * (1 - done_tensor)
            critic_sa_tensor = torch.cat((critic_state_tensor, action_tensor), dim=1)
            q_calculated = self.actor_critic.critic(critic_sa_tensor)
            critic_loss = F.mse_loss(input=q_calculated, target=q_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -torch.mean(
                self.actor_critic.critic(
                    torch.cat(
                        (
                            critic_state_tensor,
                            self.actor_critic.actor(actor_state_tensor),
                        ),
                        dim=1,
                    )
                )
            )
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            return actor_loss, critic_loss

    def save_model(self):
        torch.save(self.actor_critic.actor.state_dict(), "./logs/actor.pth")
        torch.save(self.actor_critic.critic.state_dict(), "./logs/critic.pth")

    def load_model(self):
        state_dict_actor = torch.load("./logs/actor.pth")
        state_dict_critic = torch.load("./logs/critic.pth")
        self.actor_critic.actor.load_state_dict(state_dict_actor)
        self.actor_critic.critic.load_state_dict(state_dict_critic)

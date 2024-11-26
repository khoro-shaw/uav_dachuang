import torch.nn as nn
from .actor_critic_base import ActorCriticBase


class ActorCriticQ(ActorCriticBase):
    """
    定义一个actor网络和critic网络
    actor网络输入维度是actor_state_dim，输出维度是action_dim
    critic网络充当Q网络，输入维度是critic_state_dim + action_dim，输出维度是1
    actor网络和critic网络内部隐藏层数以及节点数分别由actor_param_list和critic_param_list定义
    各种维度由环境env的dims_dict给出:（env作为参数，只是为了提供dims_dict）
    dims_dict["actor_state_dim"]
    dims_dict["critic_state_dim"]
    dims_dict["action_dim"]
    """

    def __init__(self, actor_param_list=None, critic_param_list=None, env=None):
        super(ActorCriticQ, self).__init__(actor_param_list, critic_param_list, env)
        self.critic_list[0] = nn.Linear(
            in_features=self.dims_dict["critic_state_dim"]
            + self.dims_dict["action_dim"],
            out_features=critic_param_list[0],
        )
        self.critic = nn.Sequential(*self.critic_list)

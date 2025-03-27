import torch
import torch.nn as nn
from .actor_critic_base import ActorCriticBase


class ActorCriticProbs(ActorCriticBase):
    """
    定义一个actor网络和critic网络
    actor网络输入维度是actor_state_dim，输出是2个action_dim维度的张量，分别对应动作的高斯分布的期望和标准差
    critic网络输入维度是critic_state_dim，输出维度是1
    actor网络和critic网络内部隐藏层数以及节点数分别由actor_param_list和critic_param_list定义
    各种维度由环境env的dims_dict给出:（env作为参数，只是为了提供dims_dict）
    dims_dict["actor_state_dim"]
    dims_dict["critic_state_dim"]
    dims_dict["action_dim"]
    """

    def __init__(self, actor_param_list=None, critic_param_list=None, env=None):
        super(ActorCriticProbs, self).__init__(actor_param_list, critic_param_list, env)
        self.actor = ProbsNet(param_list=actor_param_list, dims_dict=self.dims_dict)


class ProbsNet(nn.Module):
    def __init__(self, param_list=None, dims_dict=None):
        super(ProbsNet, self).__init__()
        self.fc_list = []
        self.act_list = []
        self.num = len(param_list)
        if param_list is not None and dims_dict is not None:
            for idx, num in enumerate(param_list):
                if idx == 0:
                    self.fc0 = nn.Linear(
                        in_features=dims_dict["actor_state_dim"],
                        out_features=param_list[idx],
                    )
                    self.act0 = self.get_activation()
                    self.fc_list.append(self.fc0)
                    self.act_list.append(self.act0)
                else:
                    exec(
                        f"self.fc{idx} = nn.Linear(in_features={param_list[idx - 1]}, out_features={num})"
                    )
                    exec(f"self.act{idx} = self.get_activation()")
                    exec(f"self.fc_list.append(self.fc{idx})")
                    exec(f"self.act_list.append(self.act{idx})")

            self.fc_mu = nn.Linear(
                in_features=param_list[-1], out_features=dims_dict["action_dim"]
            )
            self.fc_mu.weight.data.mul_(0.1)

            self.act_mu = self.get_activation("tanh")

            # self.fc_sigma = nn.Linear(
            #     in_features=param_list[-1], out_features=dims_dict["action_dim"]
            # )
            # self.act_sigma = self.get_activation("softplus")

        self.logstd = nn.Parameter(torch.zeros(dims_dict["action_dim"]))
        # self._initialize_weights()

    def forward(self, x):
        for i in range(self.num):
            x = self.act_list[i](self.fc_list[i](x))
        mu = self.act_mu(self.fc_mu(x))
        # std = self.act_sigma(self.fc_sigma(x))
        # return mu, std
        return mu

    # def _initialize_weights(self):
    #     # 对全连接层进行Kaiming初始化
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(
    #                 m.weight, mode="fan_in", nonlinearity="relu"
    #             )
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

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

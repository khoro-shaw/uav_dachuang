import torch.nn as nn


class ActorCriticBase(nn.Module):
    """
    定义一个actor网络和critic网络
    actor网络输入维度是actor_state_dim，输出维度是action_dim
    critic网络输入维度是critic_state_dim + action_dim，输出维度是1
    actor网络和critic网络内部隐藏层数以及节点数分别由actor_param_list和critic_param_list定义
    各种维度由环境env的dims_dict给出:（env作为参数，只是为了提供dims_dict）
    dims_dict["actor_state_dim"]
    dims_dict["critic_state_dim"]
    dims_dict["action_dim"]
    """

    def __init__(self, actor_param_list=None, critic_param_list=None, env=None):
        super(ActorCriticBase, self).__init__()
        self.actor_list = []
        self.critic_list = []
        if env is not None:
            dims_dict = env.get_dims_dict()
            if actor_param_list is not None:
                for idx, param in enumerate(actor_param_list):
                    if idx == 0:
                        self.actor_list.append(
                            nn.Linear(
                                in_features=dims_dict["actor_state_dim"],
                                out_features=param,
                            )
                        )
                        self.actor_list.append(self.get_activation())
                    else:
                        self.actor_list.append(
                            nn.Linear(
                                in_features=actor_param_list[idx - 1],
                                out_features=param,
                            )
                        )
                        self.actor_list.append(self.get_activation())

                self.actor_list.append(
                    nn.Linear(
                        in_features=actor_param_list[-1],
                        out_features=dims_dict["action_dim"],
                    )
                )
                self.actor_list.append(self.get_activation(act="tanh"))
                self.actor = nn.Sequential(*self.actor_list)
            else:
                raise ValueError("actor_param_list not defined")

            if critic_param_list is not None:
                for idx, param in enumerate(critic_param_list):
                    if idx == 0:
                        self.critic_list.append(
                            nn.Linear(
                                in_features=dims_dict["critic_state_dim"]
                                + dims_dict["action_dim"],
                                out_features=param,
                            )
                        )
                        self.critic_list.append(self.get_activation())
                    else:
                        self.critic_list.append(
                            nn.Linear(
                                in_features=critic_param_list[idx - 1],
                                out_features=param,
                            )
                        )
                        self.critic_list.append(self.get_activation())

                self.critic_list.append(
                    nn.Linear(in_features=critic_param_list[-1], out_features=1)
                )
                self.critic = nn.Sequential(*self.critic_list)
            else:
                raise ValueError("critic_param_list not defined")
        else:
            raise ValueError("env not given")

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
        else:
            print("invalid activation function")
            return None

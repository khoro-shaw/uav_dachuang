import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.runners import RunnerBase

actor_param_list = [128, 64, 16]
critic_param_list = [128, 32, 8]


params_dict = {
    # "tuple_num": 5000,
    "batch_size": 512,
    "gamma": 0.5,
    "eps": 0.2,
    "epochs": 10,
    "critic_lr": 1e-2,
    "critic_eps": 8e-2,
    "actor_lr": 1e-2,
    "actor_eps": 8e-2,
    "sigma": 1e-2,
    "tau": 5e-3,
    "lmbda": 0.95,
}

runner = RunnerBase(
    actor_param_list=actor_param_list,
    critic_param_list=critic_param_list,
    params_dict=params_dict,
)

runner.visualize()

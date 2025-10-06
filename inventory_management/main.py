import argparse
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from algo import PPO, DPPO
from env import VecInventoryManagementEnv


class Options(object):
    def __init__(self, algo_name, w_info):
        parser = argparse.ArgumentParser()
        parser.add_argument('--algo_name', type=str, default=algo_name)

        max_episode = 10000000
        parser.add_argument('--max_episode', type=int, default=max_episode)
        parser.add_argument('--warmup_episode', type=int, default=int(max_episode//4000))
        parser.add_argument('--est_interval', type=int, default=int(max_episode//1000))
        parser.add_argument('--log_interval', type=int, default=int(max_episode//1000))
        parser.add_argument('--upd_interval', type=int, default=250)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--workers', type=int, default=250)

        k_0 = 250000
        parser.add_argument('--lr_info', type=float, default=[2.5e-4, k_0, 0.99])
        parser.add_argument('--lr_q_info', type=float, default=[5e0, k_0, 0.71])
        parser.add_argument('--lr_D_info', type=float, default=[1e-3, k_0, 0.70])
        parser.add_argument('--h_info', type=float, default=[1e0, k_0, 0.14])

        parser.add_argument('--drm_n', type=int, default=100)
        parser.add_argument('--w_type', type=str, default=w_info[0])
        parser.add_argument('--w_alpha', type=list, default=w_info[1])    

        parser.add_argument('--lambda_gae_adv', type=float, default=0.95)
        parser.add_argument('--step_clip_bound', type=float, default=[0.8, 1.2])
        parser.add_argument('--episode_clip_bound', type=float, default=[0.6, 1.4])
        parser.add_argument('--vf_coef', type=float, default=0.5)
        parser.add_argument('--ent_coef', type=float, default=0.05)

        parser.add_argument('--upd_step', type=int, default=3)
        parser.add_argument('--upd_minibatch', type=int, default=5000)
        self.parser = parser

    def parse(self, seed=0, device='0'):
        args = self.parser.parse_args(args=[])
        args.seed = seed
        args.device = torch.device("cuda:" + device if torch.cuda.is_available() else "cpu")
        return args


def run(algo_name, w_info, seed=0, device=0):
    args = Options(algo_name, w_info).parse(seed=seed, device=str(device))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = VecInventoryManagementEnv(seed=seed, worker=args.workers)
    if args.algo_name == 'DPPO':
        agent = DPPO(args, env)
    elif args.algo_name == 'PPO':
        agent = PPO(args, env)
    agent.train()


if __name__ == '__main__':
    run(algo_name='PPO', w_info=['mean', [0.]])
    run(algo_name='DPPO', w_info=['mean', [0.]])
    run(algo_name='DPPO', w_info=['cpt', [0.7]])
    run(algo_name='DPPO', w_info=['wang', [-0.5]])
    run(algo_name='DPPO', w_info=['wang', [0.5]])
    run(algo_name='DPPO', w_info=['discontinuous', [0.3, 0.5, 0.7]])


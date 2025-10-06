import gym
from gym import spaces
import numpy as np


class VecInventoryManagementEnv(gym.Env):
    def __init__(self, seed=0, worker=5):
        self.T = 100

        self.l = np.array([2, 3, 5])                
        self.c = np.array([0.125, 0.1, 0.075])       
        self.h = np.array([0.20, 0.15, 0.10])       
        self.p = np.array([2.0, 1.5, 1.0, 0.5])         
        self.init_I = np.array([10, 10, 10])      
        self.init_S = np.array([10, 10, 10])

        self.lam = 10
        self.n_stage = np.size(self.l)
        self.lost_sale = True

        observation_dim = np.ones([np.max(self.l), 4 * self.n_stage])
        self.action_space = spaces.MultiDiscrete([21, 21, 21])
        self.observation_space = spaces.Box(np.array(-observation_dim*np.inf, dtype=float),
                                            np.array(observation_dim*np.inf, dtype=float), dtype=float)

        self.I, self.B, self.S, self.q, self.t, self.R = None, None, None, None, None, None
        self.state_stack = None
        self.rng = np.random.default_rng(seed)
        self.seed(seed)

        self.worker = worker

    def step(self, action: np.ndarray):
        self.t += 1
        d = self.rng.choice(7, size=self.worker) + (self.t+6) % 15
        self.q[:, self.t] += np.concatenate((d.reshape(self.worker, 1), action), axis=1)
        arrive = np.array([self.S[:, np.maximum(self.t-self.l[n], 0), n+1] for n in range(self.n_stage)]).T
        if self.lost_sale:
            diff = self.q[:, self.t, :-1] - self.I[:, self.t - 1] - arrive
            self.S[:, self.t, :-1] += self.q[:, self.t, :-1] - np.maximum(diff, 0)
        else:
            diff = self.B[:, self.t-1] + self.q[:, self.t,:-1] - self.I[:, self.t-1] - arrive
            self.S[:, self.t,:-1] += self.B[:, self.t-1] + self.q[:, self.t,:-1] - np.maximum(diff, 0)
        self.S[:, self.t,-1] += self.q[:, self.t,-1]

        self.I[:, self.t] += self.I[:, self.t-1] + arrive - self.S[:, self.t,:-1]
        if self.lost_sale:
            self.B[:, self.t] += self.q[:, self.t, :-1] - self.S[:, self.t, :-1]
        else:
            self.B[:, self.t] += self.B[:, self.t-1] + self.q[:, self.t,:-1] - self.S[:, self.t,:-1]

        done = self.t == self.T
        done = np.array([done] * self.worker)

        self.R[:, self.t,:-1] += self.p[:-1] * self.S[:, self.t,:-1] - self.p[1:] * self.S[:, self.t,1:] \
                              - self.c * self.B[:, self.t] - self.h * self.I[:, self.t]
        self.R[:, self.t,-1] += np.sum(self.R[:, self.t,:-1], axis=1)
        reward = self.R[:, self.t,-1]
        state = np.hstack((self.I[:, self.t], self.B[:, self.t], self.S[:, self.t,1:], self.q[:, self.t,:-1]))

        self.state_stack = np.concatenate((self.state_stack[1:], state.reshape(1, self.worker, -1)), axis=0)
        return self.state_stack.transpose((1, 0, 2)), reward, done, {}

    def reset(self):
        self.t = 0
        self.I = np.zeros((self.worker, self.T+1, self.n_stage))
        self.B = np.zeros((self.worker, self.T+1, self.n_stage))
        self.S = np.zeros((self.worker, self.T+1, self.n_stage+1))
        self.q = np.zeros((self.worker, self.T+1, self.n_stage+1))
        self.R = np.zeros((self.worker, self.T+1, self.n_stage+1))
        self.I[:,self.t] += self.init_I.reshape(1, -1)
        self.S[:,self.t,1:] += self.init_S.reshape(1, -1)
        state = np.hstack((self.I[:, self.t], self.B[:, self.t], self.S[:, self.t,1:], self.q[:, self.t,:-1]))
        self.state_stack = np.stack([state] * np.max(self.l))
        return self.state_stack.transpose((1, 0, 2))

    def close(self):
        return None



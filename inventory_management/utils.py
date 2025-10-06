import numpy as np
import torch
from scipy.stats import norm
from torch.distributions import Normal


def lr_lambda(k, a, b, c):
    lr = a * (b ** c) / ((b + k) ** c)
    return lr


class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.last_values = []
        self.t = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.last_values.clear()
        self.t.clear()

    def get_len(self):
        return len(self.is_terminals)
        

def indicator(x: torch.Tensor, y: torch.Tensor):
    try:
        return (y <= x.unsqueeze(1)).float()
    except IndexError:
        return torch.where(y <= x, torch.ones_like(y), torch.zeros_like(y))

    
def kernel(x, y, h=1.):
    compress = 1 / h
    tmp = compress * (x.unsqueeze(1) - y)
    return torch.where(tmp >= 0, compress * torch.exp(-tmp) / (1 + torch.exp(-tmp)) ** 2,
                       compress * torch.exp(tmp) / (1 + torch.exp(tmp)) ** 2)


class DistortionFunction:
    def __init__(self, name='mean', params=[5]):
        self.name = name
        self.params = params
        if self.name == "discontinuous":
            self.disc_points = [1 - p for p in self.params]
        else:   
            self.disc_points = []
        self.normal = Normal(torch.tensor(0.0), torch.tensor(1.0)) if self.name == 'wang' else None
    
    def __call__(self, x):
        if self.name == "mean":
            return x
        elif self.name == "cpt":
            return x**self.params[0] / (x**self.params[0] + (1-x)**self.params[0]) ** (1/self.params[0])
        elif self.name == 'wang':
            return self.normal.cdf(self.normal.icdf(x) - self.params[0])
        elif self.name == "discontinuous":
            base = (torch.exp(2*5*x)-1) / ((np.exp(5)-1) * (torch.exp(2*5*x-5)+1))
            points = 1 - torch.tensor(self.params).unsqueeze(1).to(x.device)
            tmp = torch.mean(torch.where(x <= points, torch.zeros_like(x, device=x.device), torch.ones_like(x, device=x.device)), dim=0)
            return 0.8 * base + 0.2 * tmp
        else:
            raise ValueError(f"Distortion function {self.name} not found")
    
    def prime(self, x):
        if self.name == "mean":
            return torch.ones_like(x)
        elif self.name == "cpt":
            term1 = (x**self.params[0] + (1-x)**self.params[0])
            return self.params[0] * x**(self.params[0] - 1) * term1**(-1/self.params[0]) - \
                x**self.params[0] * term1**(-1/self.params[0]) * (-self.params[0] * (1-x)**(self.params[0] - 1) + self.params[0] * x**(self.params[0] - 1)) / term1
        elif self.name == 'wang':
            pdf = torch.exp(self.normal.log_prob(self.normal.icdf(x)))
            return torch.exp(self.normal.log_prob(self.normal.icdf(x) - self.params[0])) / pdf
        elif self.name == "discontinuous":
            alpha = 5.
            tmp = 0.8 * ( 2 * alpha * (torch.exp(2 * alpha * x) + torch.exp(2 * alpha * x - 5)) / ((np.exp(alpha) - 1) * (1 + torch.exp(2 * alpha * x - alpha))**2) ) 
            points = 1 - torch.tensor(self.params).unsqueeze(1).to(x.device)
            return torch.mean(torch.where(x != points, tmp, torch.inf*torch.ones_like(x, device=x.device)), dim=0)
        else:
            raise ValueError(f"Distortion function {self.name} not found")
        

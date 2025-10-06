import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm


def kernel(x, y, h=1.):
    compress = 1 / h
    tmp = compress * (x.unsqueeze(1) - y)
    return torch.where(tmp >= 0, compress * torch.exp(-tmp) / (1 + torch.exp(-tmp)) ** 2,
                       compress * torch.exp(tmp) / (1 + torch.exp(tmp)) ** 2)


class DistortionFunction:
    def __init__(self, name='sshape', params=[5]):
        self.name = name
        self.params = params
        if self.name == "discontinuous":
            self.disc_points = [1 - p for p in self.params]
        else:   
            self.disc_points = []
    
    def __call__(self, x):
        device = x.device if torch.is_tensor(x) else None
        if self.name == 'cvar':
            tmp = x / (1 - self.params[0])
            return torch.where(tmp <= 1, tmp, torch.ones_like(x, device=device)) if torch.is_tensor(x) else np.where(tmp <= 1, tmp, np.ones_like(x))
        elif self.name == 'sshape':
            if torch.is_tensor(x):
                return (torch.exp(2*self.params[0]*x)-1) / ((np.exp(self.params[0])-1) * (torch.exp(2*self.params[0]*x-self.params[0])+1))
            else:
                return (np.exp(2*self.params[0]*x)-1) / ((np.exp(self.params[0])-1) * (np.exp(2*self.params[0]*x-self.params[0])+1))
        elif self.name == 'wang':
            if torch.is_tensor(x):
                dist = torch.distributions.Normal(loc=0, scale=1)
                return dist.cdf(dist.icdf(x) - self.params[0])
            else:
                return norm.cdf(norm.ppf(x) - self.params[0])
        elif self.name == "discontinuous":
            if torch.is_tensor(x):
                base = (torch.exp(2*5*x)-1) / ((np.exp(5)-1) * (torch.exp(2*5*x-5)+1))
                points = 1 - torch.tensor(self.params).unsqueeze(1).to(device)
                tmp = torch.mean(torch.where(x <= points, torch.zeros_like(x, device=device), torch.ones_like(x, device=device)), dim=0)
            else:
                base = (np.exp(2*5*x)-1) / ((np.exp(5)-1) * (np.exp(2*5*x-5)+1))
                points = 1 - np.array(self.params).reshape(-1, 1)
                tmp = np.mean(np.where(x <= points, np.zeros_like(x), np.ones_like(x)), axis=0)
            return 0.8 * base + 0.2 * tmp
        else:
            raise ValueError(f"Distortion function {self.name} not found")
    
    def prime(self, x):
        device = x.device if torch.is_tensor(x) else None
        if self.name == 'cvar':
            tmp = x / (1 - self.params[0])
            return torch.where(tmp <= 1, torch.tensor(1 / (1 - self.params[0]), device=device), torch.tensor(0.0, device=device)) if torch.is_tensor(x) else np.where(tmp <= 1, 1 / (1 - self.params[0]), 0)
        elif self.name == 'sshape':
            if torch.is_tensor(x):
                return 2 * self.params[0] * (torch.exp(2 * self.params[0] * x) + torch.exp(2 * self.params[0] * x - self.params[0])) / ((np.exp(self.params[0]) - 1) * (1 + torch.exp(2 * self.params[0] * x - self.params[0]))**2)
            else:
                return 2 * self.params[0] * (np.exp(2 * self.params[0] * x) + np.exp(2 * self.params[0] * x - self.params[0])) / ((np.exp(self.params[0]) - 1) * (1 + np.exp(2 * self.params[0] * x - self.params[0]))**2)
        elif self.name == 'wang':
            if torch.is_tensor(x):
                dist = torch.distributions.Normal(loc=0, scale=1)
                tmp = dist.icdf(x)
                return torch.exp(dist.log_prob(tmp - self.params[0]) - dist.log_prob(tmp))
            else:
                tmp = norm.ppf(x)
                return norm.pdf(tmp - self.params[0]) / norm.pdf(tmp)
        elif self.name == "discontinuous":
            if torch.is_tensor(x):
                tmp = 0.8 * ( 2 * 5 * (torch.exp(2 * 5 * x) + torch.exp(2 * 5 * x - 5)) / ((np.exp(5) - 1) * (1 + torch.exp(2 * 5 * x - 5))**2) ) 
                points = 1 - torch.tensor(self.params).unsqueeze(1).to(device)
                return torch.mean(torch.where(x != points, tmp, torch.inf*torch.ones_like(x, device=device)), dim=0)
            else:
                tmp = 0.8 * ( 2 * 5 * (np.exp(2 * 5 * x) + np.exp(2 * 5 * x - 5)) / ((np.exp(5) - 1) * (1 + np.exp(2 * 5 * x - 5))**2) ) 
                points = 1 - np.array(self.params).reshape(-1, 1)
                return np.mean(np.where(x != points, tmp, np.inf*np.ones_like(x)), axis=0)
        else:
            raise ValueError(f"Distortion function {self.name} not found")
    
    def envelope(self):
        from scipy.interpolate import interp1d
        u = np.linspace(0, 1, 1000)
        y = self(u)
        points = list(zip(u, y))
        hull = []
        for p in points:
            while len(hull) >= 2:
                (x0, y0), (x1, y1) = hull[-2], hull[-1]
                x2, y2 = p
                if (y2 - y0) * (x1 - x0) > (y1 - y0) * (x2 - x0):
                    hull.pop()
                else:
                    break
            hull.append(p)
        hull_points = np.array(hull)
        u_hull, y_hull = hull_points[:,0], hull_points[:,1]

        if len(hull_points) == len(points):
            envelope_prime = self.prime
        else:
            derivative_values = np.zeros_like(u_hull)
            derivative_values[-1] = (y_hull[-1] - y_hull[-2]) / (u_hull[-1] - u_hull[-2])
            for i in range(len(u_hull) - 1):
                derivative_values[i] = (y_hull[i + 1] - y_hull[i]) / (u_hull[i + 1] - u_hull[i])
            envelope_prime = interp1d(u_hull, derivative_values, kind='previous', fill_value='extrapolate')
        return envelope_prime

    def real_solution(self):
        envelope_prime = self.envelope()    
        z_prime = np.linspace(0, 1, 10000)[1:-1]
        g_vals = envelope_prime(1 - z_prime)
        integrand = (g_vals - 1)**2
        C = 1 / np.sqrt(np.trapz(integrand, z_prime))
        def q(z):
            z = np.asarray(z)
            g_val = envelope_prime(1 - z)
            return C * (g_val - 1)
        return q


class MixGaussian(nn.Module):
    def __init__(self, raw_mu_bound, raw_log_sigma_bound, raw_weight_bound, mix_n, seed):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        if len(raw_mu_bound) == 2:
            raw_mu = torch.empty(mix_n).uniform_((5 * raw_mu_bound[0] + 3 * raw_mu_bound[1]) / 8, 
                                                (3 * raw_mu_bound[0] + 5 * raw_mu_bound[1]) / 8, generator=self.rng)
        else:
            # easier setting, only used when all baselines fail
            # but it is ok for all proposed methods
            raw_mu = torch.where(torch.rand(mix_n, generator=self.rng) < 0.5, 
                                ((7 * raw_mu_bound[0] + 1 * raw_mu_bound[1]) / 8)*torch.ones(mix_n), 
                                ((1 * raw_mu_bound[0] + 7 * raw_mu_bound[1]) / 8)*torch.ones(mix_n))
        raw_log_sigma = torch.empty(mix_n).uniform_((5 * raw_log_sigma_bound[0] + 3 * raw_log_sigma_bound[1]) / 8,
                                                   (3 * raw_log_sigma_bound[0] + 5 * raw_log_sigma_bound[1]) / 8, generator=self.rng)
        raw_weight = torch.empty(mix_n).uniform_((5 * raw_weight_bound[0] + 3 * raw_weight_bound[1]) / 8,
                                                (3 * raw_weight_bound[0] + 5 * raw_weight_bound[1]) / 8, generator=self.rng)
        self.raw_mu = torch.nn.Parameter(raw_mu, requires_grad=True)
        self.raw_log_sigma = torch.nn.Parameter(raw_log_sigma, requires_grad=True)
        self.raw_weight = torch.nn.Parameter(raw_weight, requires_grad=True)
        self.mu, self.sigma, self.weight = None, None, None

        self.raw_mu_bound = raw_mu_bound
        self.raw_log_sigma_bound = raw_log_sigma_bound
        self.raw_weight_bound = raw_weight_bound
    
    def to(self, device):
        super().to(device)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(self.rng.initial_seed())
        return self

    def forward(self, x):
        return self.pdf(x)
    
    def reset(self):
        self.mu, self.sigma, self.weight = None, None, None

    def preprocess(self):
        self.weight = torch.softmax(self.raw_weight, dim=0)
        mu_mix = torch.sum(self.weight * self.raw_mu)
        mu = self.raw_mu - mu_mix
        sigma = torch.exp(self.raw_log_sigma)
        sigma_mix = torch.sqrt(torch.sum(self.weight * (sigma ** 2 + mu ** 2)))
        self.mu = mu / sigma_mix
        self.sigma = sigma / sigma_mix

    def pdf(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.raw_mu.device)
        self.preprocess()
        x_expanded = x.unsqueeze(-1)
        coef = 1 / (self.sigma * np.sqrt(2.0 * np.pi))
        exponent = torch.exp(-0.5 * ((x_expanded - self.mu) / self.sigma) ** 2)
        pdf_val = torch.sum(self.weight * coef * exponent, dim=-1)
        self.reset()
        return pdf_val

    def cdf(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.preprocess()
        x = x.to(self.raw_mu.device)
        x_expanded = x.unsqueeze(-1)
        z = (x_expanded - self.mu) / (self.sigma * np.sqrt(2.0))
        cdf_val = torch.sum(self.weight * 0.5 * (1 + torch.erf(z)), dim=-1)
        self.reset()
        return cdf_val

    def sample(self, n_samples=1):
        with torch.no_grad():
            self.preprocess()
            components = torch.multinomial(self.weight, n_samples, replacement=True, generator=self.rng)
            mu = self.mu[components].detach()
            std = self.sigma[components].detach()
            samples = torch.normal(mu, std, generator=self.rng)
            self.reset()
            return samples

    def ppf(self, z, tol=1e-6, max_iter=100):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z = z.to(self.raw_mu.device)
        lower = torch.full(z.shape, -100.0, device=self.raw_mu.device)
        upper = torch.full(z.shape, 100.0, device=self.raw_mu.device)
        mid = (lower + upper) / 2.0
        iteration = 0
        while iteration < max_iter:
            mid_cdf = self.cdf(mid)
            within_tol = torch.abs(mid_cdf - z) < tol
            if torch.all(within_tol):
                break
            lower = torch.where(mid_cdf < z, mid, lower)
            upper = torch.where(mid_cdf > z, mid, upper)
            mid = (lower + upper) / 2.0
            iteration += 1
        return mid

    def project_para(self):
        self.raw_mu.data.clamp_(self.raw_mu_bound[0], self.raw_mu_bound[1])
        self.raw_log_sigma.data.clamp_(self.raw_log_sigma_bound[0], self.raw_log_sigma_bound[1])
        self.raw_weight.data.clamp_(self.raw_weight_bound[0], self.raw_weight_bound[1])


def extrapolation(z, activate=True):
    if activate:
        return torch.cat([z[0] - (z[1] - z[0]).unsqueeze(0), z, z[-1] + (z[-1] - z[-2]).unsqueeze(0)])
    else:
        return z


def lr_lambda(k, a, b, c):
    lr = a * (b ** c) / ((b + k) ** c)
    return lr


def indicator(x: torch.Tensor, y: torch.Tensor):
    try:
        return (y <= x.unsqueeze(1)).float()
    except IndexError:
        return torch.where(y <= x, torch.ones_like(y), torch.zeros_like(y))  


def compute_drm(dist, w, n_nodes: int = 1024, eps: float = 1e-6):
    device = dist.raw_mu.device
    z = torch.linspace(eps, 1 - eps, n_nodes + 1, device=device)
    q = dist.ppf(z) 
    wz = w(1 - z) 
    val = -torch.sum(0.5 * (q[:-1] + q[1:]) * ( wz[1:] - wz[:-1]))
    return val.item()


def compute_ppf_loss(func1, func2, n=10000, bounds=[1e-6, 1-1e-6]):
    z = torch.linspace(bounds[0], bounds[1], n)
    f1_vals = func1(z)
    f2_vals = func2(z)
    if isinstance(f1_vals, torch.Tensor):
        f1_vals = f1_vals.cpu()
    if isinstance(f2_vals, torch.Tensor):
        f2_vals = f2_vals.cpu()
    diff_squared = (f1_vals - f2_vals) ** 2
    return torch.trapz(diff_squared, z)



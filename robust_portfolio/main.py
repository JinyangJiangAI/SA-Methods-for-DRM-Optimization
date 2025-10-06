import numpy as np
from copy import deepcopy
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.func import vmap, grad, functional_call
from collections import OrderedDict
from tqdm import tqdm
import json
from types import SimpleNamespace
import random
import warnings
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import *


def compute_loss(model, params, x, ind):
    log_pdf = torch.log(functional_call(model, params, (x,)))
    return torch.mean(ind * log_pdf)


def run(args, seed=0, show_progress=True):
    if show_progress: 
        pbar = tqdm(total=args.max_iter, desc=f'Seed {seed}', position=0, leave=True)
    random.seed(seed)
    device = torch.device('cuda:'+str(seed%torch.cuda.device_count()) if args.device == 'cuda' else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    w = DistortionFunction(name=args.distortion_name, params=args.distortion_params)
    solu_ppf = w.real_solution()

    z = torch.linspace(0, 1, args.drm_n+3)[1:-1]
    z = z.to(device)
    z_ = (z[1:] + z[:-1]) / 2

    z = z ** args.grid_pow
    z_ = z_ ** args.grid_pow
    unif_grid = True if args.grid_pow == 1.0 else False

    drm_weight = -w.prime(1 - z_)
    drm_weight_ = w(1-z[1:]) - w(1-z[:-1])

    if args.alg == "hybrid-form":
        mask = torch.zeros_like(z_, dtype=torch.bool, device=device)
        mask[torch.searchsorted(z[1:], torch.tensor(w.disc_points, device=device), right=True)] = True
        mask = torch.flip(mask, dims=[0])
    elif args.alg == "dm-form":
        mask = torch.ones_like(z_, dtype=torch.bool, device=device)

    dist = MixGaussian(args.raw_mu_bound, args.raw_log_sigma_bound, args.raw_weight_bound, args.mix_n, seed).to(device)

    optimizer_theta = torch.optim.SGD(dist.parameters(), lr=args.lr_theta_info[0])
    scheduler_theta = LambdaLR(optimizer_theta, lr_lambda=lambda k: lr_lambda(k, 1, args.lr_theta_info[1], args.lr_theta_info[2]))
    if "batching" not in args.alg:
        q = torch.nn.Parameter(torch.zeros_like(z).clone(), requires_grad=True).to(device)
        q_ = torch.nn.Parameter(torch.zeros_like(z_).clone(), requires_grad=True).to(device)
        optimizer_q = torch.optim.SGD([q, q_], lr=args.lr_q_info[0])
        scheduler_q = LambdaLR(optimizer_q, lr_lambda=lambda k: lr_lambda(k, 1, args.lr_q_info[1], args.lr_q_info[2]))
    if args.alg in ["dm-form", "hybrid-form"]:
        if torch.sum(mask) > 0:
            D_dict = OrderedDict()
            for name, p in dist.named_parameters():
                D_dict[name] = torch.nn.Parameter(torch.zeros((torch.sum(mask),) + p.shape, device=device, dtype=p.dtype, requires_grad=True) )
            optimizer_D = torch.optim.SGD(D_dict.values(), lr=args.lr_D_info[0])
            scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda k: lr_lambda(k, 1, args.lr_D_info[1], args.lr_D_info[2]))  
            band_width = lambda k: lr_lambda(k, args.h_info[0], args.h_info[1], args.h_info[2])

    # warming up
    X = dist.sample(args.drm_n+3).to(device)
    if "batching" not in args.alg:
        q.data, q_.data = torch.quantile(X, z).to(device), torch.quantile(X, z_).to(device)
    else:
        q, q_ = torch.quantile(X, z).to(device), torch.quantile(X, z_).to(device)
    if args.alg in ["dm-form", "hybrid-form"]:
        if torch.sum(mask) > 0:   
            with torch.no_grad():
                ind = indicator(q_[mask], X).to(device)
                kernel_value = torch.mean(kernel(q_[mask], X, 1.).to(device), axis=1)
            params = dict(dist.named_parameters())
            per_ind_grads = vmap(grad(lambda params, x, ind: compute_loss(dist, params, x, ind)), in_dims=(None, None, 0))(params, X, ind)
            for name, p in dist.named_parameters(): 
                D_dict[name].data = -(per_ind_grads[name] / kernel_value.unsqueeze(1))

    with torch.no_grad():
        if show_progress:
            pbar.set_postfix({'DRM': f'{compute_drm(dist, w):.4f}', 'Loss_ppf': f'{compute_ppf_loss(dist.ppf, solu_ppf).item():.4f}'})

    # optimize
    for i in range(args.max_iter):
        if show_progress:
            pbar.update(1)
        optimizer_theta.zero_grad()

        X = dist.sample(args.sample_n).to(device)
        
        if "batching" not in args.alg:
            optimizer_q.zero_grad()
            q.grad = -torch.mean(z.unsqueeze(1) - indicator(q, X), dim=1).to(device)
            q_.grad = -torch.mean(z_.unsqueeze(1) - indicator(q_, X), dim=1).to(device)
        else:
            q, q_ = torch.quantile(X, z).to(device), torch.quantile(X, z_).to(device)

        if args.alg == "qf-form":
            with torch.no_grad():
                ind = indicator(extrapolation(q_, unif_grid), X).to(device) - extrapolation(z_, unif_grid).unsqueeze(1).to(device)
            ext_q = extrapolation(q, unif_grid).to(device)
            log_prob = torch.log(dist.pdf(X)).to(device)
            obj_theta = -torch.sum(torch.mean(log_prob * ind, dim=1) * extrapolation(drm_weight, unif_grid) * (ext_q[1:] - ext_q[:-1]).detach()).to(device)
            obj_theta.backward()
        elif args.alg == "dm-form":
            optimizer_D.zero_grad()
            with torch.no_grad():
                ind = indicator(q_, X).to(device) - z_.unsqueeze(1).to(device)
                kernel_value = torch.mean(kernel(q_, X, band_width(i)).to(device), axis=1)
            params = dict(dist.named_parameters())
            per_ind_grads = vmap(grad(lambda params, x, ind: compute_loss(dist, params, x, ind)), in_dims=(None, None, 0))(params, X, ind)
            for name, p in dist.named_parameters(): 
                D_dict[name].grad = -(per_ind_grads[name] - D_dict[name] * kernel_value.unsqueeze(1)).to(device)
                p.grad = - torch.sum(extrapolation(D_dict[name].data, unif_grid) * extrapolation(drm_weight_.unsqueeze(1), unif_grid), dim=0).to(device) 
        elif args.alg == "hybrid-form":
            with torch.no_grad():
                mask_pad = torch.cat([torch.tensor([False], device=device), mask, torch.tensor([False], device=device)])
                ind = indicator(extrapolation(q_, unif_grid), X).to(device) - extrapolation(z_, unif_grid).unsqueeze(1).to(device)
            ext_q = extrapolation(q, unif_grid).to(device)
            log_prob = torch.log(dist.pdf(X)).to(device)
            tmp = torch.mean(log_prob * ind, dim=1) * extrapolation(drm_weight, unif_grid) * (ext_q[1:] - ext_q[:-1]).detach().to(device)
            obj_theta = -torch.sum(tmp[~mask_pad])
            obj_theta.backward()
            if torch.sum(mask) > 0:
                optimizer_D.zero_grad()
                with torch.no_grad():
                    ind = ind[mask_pad]
                    kernel_value = torch.mean(kernel(q_[mask], X, band_width(i)).to(device), axis=1)
                params = dict(dist.named_parameters())
                per_ind_grads = vmap(grad(lambda params, x, ind: compute_loss(dist, params, x, ind)), in_dims=(None, None, 0))(params, X, ind)

                for name, p in dist.named_parameters(): 
                    D_dict[name].grad = -(per_ind_grads[name] - D_dict[name] * kernel_value.unsqueeze(1)).to(device)
                    p.grad += - torch.sum(D_dict[name].data * drm_weight_[mask].unsqueeze(1), dim=0).to(device)
        else:
            with torch.no_grad():
                ind = indicator(extrapolation(q_, unif_grid), X).to(device)
            ext_q = extrapolation(q, unif_grid).to(device)
            log_prob = torch.log(dist.pdf(X)).to(device)
            obj_theta = -torch.sum(torch.mean(log_prob * ind, dim=1) * extrapolation(drm_weight, unif_grid) * (ext_q[1:] - ext_q[:-1]).detach()).to(device)
            obj_theta.backward()

        optimizer_theta.step()
        scheduler_theta.step()
        dist.project_para()
        if "batching" not in args.alg:        
            optimizer_q.step()
            scheduler_q.step()
        if args.alg in ["dm-form", "hybrid-form"] and torch.sum(mask) > 0:
            optimizer_D.step()
            scheduler_D.step()

        with torch.no_grad():
            if ((i + 1) % (args.max_iter // args.plot_n) == 0) and show_progress:
                pbar.set_postfix({'DRM': f'{compute_drm(dist, w):.4f}', 'Loss_ppf': f'{compute_ppf_loss(dist.ppf, solu_ppf).item():.4f}'})

    if show_progress:
        pbar.close()


def load_args(distortion_name="sshape", alg="dm-form"):
    path=f"configs/{distortion_name}.json"
    with open(path, "r") as f:
        cfg = json.load(f)
    args = SimpleNamespace(**cfg)
    args.alg = alg
    if "batching-" in alg:
        factor = int(alg.split("-")[-1])
        args.max_iter = int(args.max_iter // factor)
        args.lr_theta_info[0] = args.lr_theta_info[0] * factor
        args.lr_theta_info[1] = int(args.lr_theta_info[1] // factor)
        args.sample_n = args.sample_n * factor
    return args


if __name__ == "__main__":
    run(args=load_args(distortion_name="sshape", alg="dm-form"))
    run(args=load_args(distortion_name="sshape", alg="qf-form"))
    run(args=load_args(distortion_name="sshape", alg="batching"))
    run(args=load_args(distortion_name="sshape", alg="batching-5"))
    run(args=load_args(distortion_name="sshape", alg="batching-10"))

    # run(args=load_args(distortion_name="wang", alg="dm-form"))
    # run(args=load_args(distortion_name="wang", alg="qf-form"))
    # run(args=load_args(distortion_name="wang", alg="batching"))
    # run(args=load_args(distortion_name="wang", alg="batching-5"))
    # run(args=load_args(distortion_name="wang", alg="batching-10"))

    # run(args=load_args(distortion_name="cvar", alg="dm-form"))
    # run(args=load_args(distortion_name="cvar", alg="qf-form"))
    # run(args=load_args(distortion_name="cvar", alg="batching"))
    # run(args=load_args(distortion_name="cvar", alg="batching-5"))
    # run(args=load_args(distortion_name="cvar", alg="batching-10"))

    # run(args=load_args(distortion_name="discontinuous", alg="dm-form"))
    # run(args=load_args(distortion_name="discontinuous", alg="hybrid-form"))
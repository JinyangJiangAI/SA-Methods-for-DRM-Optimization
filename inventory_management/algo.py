import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from torch.func import vmap, grad, functional_call
from copy import deepcopy

from model import ActorDiscrete, Critic
from utils import Memory, indicator, kernel, DistortionFunction, lr_lambda


class Base(object):
    def __init__(self, args, env):
        self.device = args.device
        self.workers = args.workers
        self.gamma = args.gamma
        self.max_episode = args.max_episode

        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.upd_interval = args.upd_interval
        self.warmup_episode = args.warmup_episode

        self.env = env
        self.state_dim = list(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.nvec
        self.actor = ActorDiscrete(self.state_dim, self.action_dim).to(self.device)

        self.optimizer = Adam(self.actor.parameters(), args.lr_info[0], eps=1e-5)
        self.MSELoss = torch.nn.MSELoss()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda k: lr_lambda(k, 1, args.lr_info[1], args.lr_info[2]))

        self.memory = Memory()

    def log(self, data, i_episode):
        data = np.array(data, dtype=np.float32)
        avg = np.mean(data)
        q_info = torch.quantile(torch.tensor(data), torch.tensor([0.1, 0.3,  0.5,  0.7, 0.9]))

        print(f'Epi:{i_episode:7d} || AVG:{avg:9.3f} | Q01:{q_info[0]:9.3f} | Q03:{q_info[1]:9.3f} | Q05:{q_info[2]:9.3f} | Q07:{q_info[3]:9.3f} | Q09:{q_info[4]:9.3f}\n')

    def train(self):
        disc_epi_rewards = []
        for i_episode in range(0, self.max_episode, self.workers):
            if i_episode == 0 and self.warmup_episode > 0:
                self.warm_up(self.warmup_episode)

            disc_epi_reward, disc_factor, state = np.zeros((self.workers,)), 1, self.env.reset()
            while True:
                state = np.array(state).transpose((0, 2, 1))
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done[0]:
                    disc_epi_rewards.extend(disc_epi_reward)
                    break

            for i in range(self.workers):
                i_epi = i_episode + i
                if i_epi % self.upd_interval == 0:
                    self.update(i_epi)
                    self.memory.clear()
                if i_epi % self.log_interval == 0 and i_epi != 0:
                    self.log(disc_epi_rewards[-self.est_interval:], i_epi)
        
    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprobs = torch.sum(dist.log_prob(action).detach(),dim=-1)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprobs)
        return action.detach().data.cpu().numpy()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = torch.sum(dist.log_prob(action),dim=-1)
        dist_entropy = torch.sum(dist.entropy(),dim=-1)
        return action_logprobs, dist_entropy

    def compute_reward2go(self):
        memory_len = self.memory.get_len()
        reward2go = np.zeros(memory_len, dtype=float)
        pre_r_sum = 0
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                pre_r_sum = 0
            reward2go[i] = self.memory.rewards[i] + self.gamma * pre_r_sum
            pre_r_sum = reward2go[i]
        reward2go = torch.from_numpy(reward2go).to(self.device).float()
        return reward2go

    def memory_reshape(self):
        memory = Memory()
        for i in range(self.workers):
            memory.actions.extend([x[i] for x in self.memory.actions])
            memory.states.extend([x[i] for x in self.memory.states])
            memory.logprobs.extend([x[i] for x in self.memory.logprobs])
            memory.rewards.extend([x[i] for x in self.memory.rewards])
            memory.is_terminals.extend([x[i] for x in self.memory.is_terminals])
            memory.values.extend([x[i] for x in self.memory.values])
            memory.last_values.extend([x[i] for x in self.memory.last_values])
        self.memory = memory

    def warm_up(self, max_episode):
        pass

    def update(self, i_episode):
        pass


class PPO(Base):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.step_clip_bound = args.step_clip_bound
        self.lambda_gae_adv = args.lambda_gae_adv
        self.vf_coef = args.vf_coef
        self.ent_coef = args.ent_coef
        self.upd_minibatch = args.upd_minibatch
        self.upd_step = args.upd_step
        self.critic = Critic(self.state_dim).to(self.device)
        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), args.lr_info[0], eps=1e-5)

    def warm_up(self, max_episode):
        env = deepcopy(self.env)
        disc_epi_rewards = np.zeros((max_episode, ), dtype=float)
        for i_episode in range(0, max_episode, self.workers):
            disc_epi_reward, disc_factor, state = np.zeros((self.workers,)), 1, env.reset()
            while True:
                workers = np.minimum(max_episode - i_episode, self.workers)
                state = np.array(state).transpose((0, 2, 1))
                action = self.choose_action(state)
                state, reward, done, _ = env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                if done[0]:
                    disc_epi_rewards[i_episode:i_episode + workers] += disc_epi_reward[:workers]
                    break
        self.log(disc_epi_rewards, 0)
        self.memory.clear()

    def train(self):
        disc_epi_rewards = []
        for i_episode in range(0, self.max_episode, self.workers):
            if i_episode == 0 and self.warmup_episode > 0:
                self.warm_up(self.warmup_episode)

            disc_epi_reward, disc_factor, state = np.zeros((self.workers,)), 1, self.env.reset()
            while True:
                state = np.array(state).transpose((0, 2, 1))
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done[0]:
                    state = np.array(state).transpose((0, 2, 1))
                    last_value = self.critic(torch.from_numpy(state).float().to(self.device)).detach().data.cpu().numpy()
                    self.memory.last_values.append(last_value)
                    disc_epi_rewards.extend(disc_epi_reward)
                    break

            for i in range(self.workers):
                i_epi = i_episode + i
                if i_epi % self.upd_interval == 0:
                    self.update(i_epi)
                    self.memory.clear()
                if i_epi % self.log_interval == 0 and i_epi != 0:
                    self.log(disc_epi_rewards[-self.est_interval:], i_epi)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprobs = torch.sum(dist.log_prob(action).detach(),dim=-1)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprobs)
        value = self.critic(state).detach().data.cpu().numpy()
        self.memory.values.append(value)
        return action.detach().data.cpu().numpy()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = torch.sum(dist.log_prob(action),dim=-1)
        dist_entropy = torch.sum(dist.entropy(),dim=-1)
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
    def compute_reward2go_gae(self):
        memory_len = self.memory.get_len()
        reward2go = np.zeros(memory_len, dtype=float)
        advantage_value = np.zeros(memory_len, dtype=float)
        pre_r_sum = 0
        pre_adv_v = 0
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                pre_r_sum = 0
                pre_adv_v = self.memory.last_values.pop()
            reward2go[i] = self.memory.rewards[i] + self.gamma * pre_r_sum
            pre_r_sum = reward2go[i]
            advantage_value[i] = self.memory.rewards[i] + self.gamma * pre_adv_v - self.memory.values[i]
            pre_adv_v = self.memory.values[i] + advantage_value[i] * self.lambda_gae_adv
        reward2go = torch.from_numpy(reward2go).to(self.device).float()
        advantage_value = torch.from_numpy(advantage_value).to(self.device).float()
        return reward2go, advantage_value
    
    def update(self, i_episode):
        self.memory_reshape()

        reward2go, advantage_value = self.compute_reward2go_gae()
        advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e-6)

        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        n_data = len(self.memory.states)
        shuffle_idx = np.arange(n_data)
        for _ in range(self.upd_step):
            np.random.shuffle(shuffle_idx)
            for i in range(n_data//self.upd_minibatch):
                if i == n_data//self.upd_minibatch - 1:
                    idx = shuffle_idx[self.upd_minibatch*i: n_data-1]
                else:
                    idx = shuffle_idx[self.upd_minibatch*i: self.upd_minibatch*(i+1)]
                self.optimizer.zero_grad()
                logprobs, state_values, dist_entropy = self.evaluate(old_states[idx], old_actions[idx])
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantage_value[idx]
                surr2 = torch.clamp(ratios, self.step_clip_bound[0], self.step_clip_bound[1]) * advantage_value[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss =  self.MSELoss(state_values, reward2go[idx]).mean()
                dist_entropy = dist_entropy.mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * dist_entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=.5)
                self.optimizer.step()

        for _ in range(sum(self.memory.is_terminals)):
            self.scheduler.step()


class DPPO(Base):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.ent_coef = args.ent_coef
        self.episode_clip_bound = args.episode_clip_bound

        self.drm_n = args.drm_n
        self.w = DistortionFunction(name=args.w_type, params=args.w_alpha)  
        self.alpha = torch.linspace(0, 1, self.drm_n+3)[1:-1].to(self.device)
        self.alpha_ = (self.alpha[1:] + self.alpha[:-1]) / 2.
        self.drm_weight = -self.w.prime(1 - self.alpha_).to(self.device)
        self.drm_weight_ = self.w(1-self.alpha[1:]) - self.w(1-self.alpha[:-1]).to(self.device)

        self.mask = torch.zeros_like(self.alpha_, dtype=torch.bool, device=self.device)
        self.mask[torch.searchsorted(self.alpha[1:], torch.tensor(self.w.disc_points, device=self.device), right=True)] = True
        self.mask = torch.flip(self.mask, dims=[0])

        self.upd_step = args.upd_step
        self.upd_minibatch = args.upd_minibatch

        self.q_est = nn.Parameter(torch.zeros_like(self.alpha).clone(), requires_grad=True).to(self.device)
        self.q_est_ = nn.Parameter(torch.zeros_like(self.alpha_).clone(), requires_grad=True).to(self.device)
        self.q_optimizer = Adam([self.q_est, self.q_est_], args.lr_q_info[0], eps=1e-5)
        self.q_scheduler = LambdaLR(self.q_optimizer, lr_lambda=lambda k: lr_lambda(k, 1, args.lr_q_info[1], args.lr_q_info[2]))
    
        if torch.sum(self.mask) > 0:
            self.D = OrderedDict()
            for name, p in self.actor.named_parameters():
                self.D[name] = torch.nn.Parameter(torch.zeros((torch.sum(self.mask),) + p.shape, device=self.device, dtype=p.dtype, requires_grad=True) )
            self.D_optimizer = Adam(self.D.values(), args.lr_D_info[0], eps=1e-5)
            self.D_scheduler = LambdaLR(self.D_optimizer, lr_lambda=lambda k: lr_lambda(k, 1, args.lr_D_info[1], args.lr_D_info[2]))  
            self.band_width = lambda k: lr_lambda(k, args.h_info[0], args.h_info[1], args.h_info[2])

    def warm_up(self, max_episode):
        env = deepcopy(self.env)
        disc_epi_rewards = np.zeros((max_episode, ), dtype=float)
        for i_episode in range(0, max_episode, self.workers):
            disc_epi_reward, disc_factor, state = np.zeros((self.workers,)), 1, env.reset()
            while True:
                workers = np.minimum(max_episode - i_episode, self.workers)
                state = np.array(state).transpose((0, 2, 1))
                action = self.choose_action(state)
                state, reward, done, _ = env.step(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done[0]:
                    disc_epi_rewards[i_episode:i_episode + workers] += disc_epi_reward[:workers]
                    break
                    
        self.log(disc_epi_rewards, 0)
        
        tmp = torch.tensor(disc_epi_rewards).float().to(self.device)
        self.q_est.data = torch.quantile(tmp, self.alpha)
        self.q_est_.data = torch.quantile(tmp, self.alpha_)
        self.memory_reshape()
        if torch.sum(self.mask) > 0:
            disc_reward, _ = self.compute_discounted_epi_reward()
            old_states = torch.stack(self.memory.states).detach()
            old_actions = torch.stack(self.memory.actions).detach()
            self.optimizer.zero_grad()
            with torch.no_grad():
                ind = indicator(self.q_est_[self.mask].detach(), disc_reward.to(self.device))
                kernel_value = torch.mean(kernel(self.q_est_[self.mask], tmp, self.band_width(0)).to(self.device), axis=1)
            params = dict(self.actor.named_parameters())
            per_ind_grads = vmap(grad(lambda params, s, a, w: self.compute_loss(params, s, a, w)),
                                in_dims=(None, None, None, 0))(params, old_states, old_actions, ind)
            for name, _ in self.actor.named_parameters(): 
                self.D[name].data = -(per_ind_grads[name] / kernel_value.view(-1, *([1] * (len(self.D[name].shape) - 1))))
        self.memory.clear()

    def update(self, i_episode):
        self.memory_reshape()
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()
        disc_reward, disc_reward_short = disc_reward.to(self.device), disc_reward_short.to(self.device)
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        end_idices = [i for i, x in enumerate(self.memory.is_terminals) if x]
        n_episode = len(end_idices)
        n_batch = int(np.ceil(len(disc_reward)//self.upd_minibatch))
        batch_episode = int(n_episode // n_batch)

        for _ in range(self.upd_step):
            episode_idices = np.arange(n_episode)
            np.random.shuffle(episode_idices)
            for ii in range(n_batch): 
                episode_list = episode_idices[ii*batch_episode: (ii+1)*batch_episode]
                step_list, step_mask = [], [0]
                for j in episode_list:
                    tmp = list(range(end_idices[j-1]+1, end_idices[j]+1)) if j != 0 else list(range(end_idices[j]+1))
                    step_list = step_list + tmp
                    step_mask.append(len(tmp)+step_mask[-1])

                self.optimizer.zero_grad()
                self.q_optimizer.zero_grad()
                if torch.sum(self.mask) > 0:
                    self.D_optimizer.zero_grad()

                with torch.no_grad():
                    ind = indicator(self.q_est, disc_reward_short[episode_list])
                    ind_ = indicator(self.q_est_, disc_reward_short[episode_list])

                logprobs, dist_entropy = self.evaluate(old_states[step_list], old_actions[step_list])
                dist_entropy = dist_entropy.mean()
                step_log_ratios = logprobs - old_logprobs[step_list]
                log_ratios = torch.stack([torch.sum(step_log_ratios[step_mask[t]:step_mask[t+1]]) for t in range(batch_episode)])
                ratios = torch.exp(log_ratios)

                flag = ((ratios > self.episode_clip_bound[0]) & (ratios < self.episode_clip_bound[1]))
                if torch.sum(flag) > 0:
                    tmp1 = (- self.alpha.unsqueeze(1) + ind)*ratios
                    tmp2 = (- self.alpha_.unsqueeze(1) + ind_)*ratios
                    self.q_est.grad = torch.mean(tmp1[:,flag], dim=1).detach()
                    self.q_est_.grad = torch.mean(tmp2[:,flag], dim=1).detach()
                    tmp3 = ratios * (ind_ - self.alpha_.unsqueeze(1))
                    tmp = torch.mean(tmp3[:,flag], dim=1) * self.drm_weight * (self.q_est[1:] - self.q_est[:-1]).detach()
                    obj_theta = - torch.sum(tmp[~self.mask]) - self.ent_coef * dist_entropy
                    obj_theta.backward()
                    if torch.sum(self.mask) > 0:
                        with torch.no_grad():
                            valid_step_list = sum([step_list[step_mask[t]:step_mask[t+1]] for t in range(batch_episode) if flag[t]], [])
                            extended_ratios = torch.cat([ratios[t]*torch.ones((step_mask[t+1]-step_mask[t]),device=self.device) for t in range(batch_episode) if flag[t]])
                            ind_step = (indicator(self.q_est_[self.mask], disc_reward[valid_step_list]) - self.alpha_[self.mask].unsqueeze(1)) * extended_ratios
                            kernel_value = torch.mean(kernel(self.q_est_[self.mask], disc_reward_short[episode_list[flag.cpu().numpy()]], self.band_width(i_episode))*ratios[flag], axis=1)
                        params = dict(self.actor.named_parameters())
                        per_ind_grads = vmap(grad(lambda params, s, a, w: self.compute_loss(params, s, a, w)),
                                            in_dims=(None, None, None, 0))(params, old_states[valid_step_list], old_actions[valid_step_list], ind_step)
                        for name, p in self.actor.named_parameters(): 
                            self.D[name].data = -(per_ind_grads[name] - self.D[name] * kernel_value.view(-1, *([1] * (len(self.D[name].shape) - 1))))
                            p.grad += - torch.sum(self.D[name].data * self.drm_weight_[self.mask].view(-1, *([1] * (len(self.D[name].shape) - 1))), dim=0)
                else:
                    obj_theta = - self.ent_coef * dist_entropy
                    obj_theta.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=.5)

                self.optimizer.step()
                self.q_optimizer.step()
                if torch.sum(self.mask) > 0:
                    self.D_optimizer.zero_grad()

        for _ in range(sum(self.memory.is_terminals)):
            self.scheduler.step()
            self.q_scheduler.step()
            if torch.sum(self.mask) > 0:
                self.D_scheduler.step()

    def compute_discounted_epi_reward(self):
        memory_len = self.memory.get_len()
        disc_reward, disc_reward_short = np.zeros(memory_len, dtype=float), []
        pre_r_sum, p1, p2 = 0, 0, 0
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                if p1 > 0:
                    disc_reward[memory_len - p1: memory_len - p2] += pre_r_sum
                    disc_reward_short.insert(0, pre_r_sum)
                pre_r_sum, p2 = 0, p1
            pre_r_sum = self.memory.rewards[i] + self.gamma * pre_r_sum
            p1 += 1
        disc_reward[memory_len - p1: memory_len - p2] += pre_r_sum
        disc_reward_short.insert(0, pre_r_sum)
        disc_reward = torch.from_numpy(disc_reward).to(self.device).float()
        disc_reward_short = torch.tensor(disc_reward_short).to(self.device).float()
        return disc_reward, disc_reward_short

    def compute_loss(self, params, states, actions, ind):
        action_probs = functional_call(self.actor, params, (states,))
        dist = Categorical(action_probs)
        action_logprobs = torch.sum(dist.log_prob(actions), dim=-1)
        return torch.mean(ind * action_logprobs)

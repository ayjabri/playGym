#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:46:25 2020
@author: Ayman Jabri

Continuous action space using Actor Critic
Env: MountainCarContinuous
Steps:
    1- Build NN with three heads to reutrn mu, var and value
    2-

"""

import gym
import ptan
import torch
import torch.nn as nn
import numpy as np
from math import pi

# =============================================================================
# A2C NN with 3 heards
# =============================================================================
class A2C(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()

        self.base = nn.Sequential(nn.Linear(obs_size, hid_size),
                                  nn.ReLU())
        self.value= nn.Linear(hid_size, 1)
        self.mu = nn.Sequential(nn.Linear(hid_size, act_size),
                                nn.Tanh())
        self.var = nn.Sequential(nn.Linear(hid_size, act_size),
                                 nn.Softplus())

    def forward(self, x)        :
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)

# =============================================================================
# Custom A2C agent based on standard ptan.agent class
# =============================================================================
class A2C_Agent(ptan.agent.BaseAgent):
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def __call__(self, obs, agent_states):
        obs_v = ptan.agent.float32_preprocessor(obs).to(self.device)
        mu_v, var_v, _ = self.model(obs_v)
        mu = mu_v.data.cpu().numpy()
        var = var_v.data.cpu().numpy()
        sigma = np.sqrt(var)
        action = np.random.normal(mu, sigma)
        action = np.clip(action, -1, 1)
        return action, agent_states

# =============================================================================
# Critic playing N episodes, returning rewards and steps
# =============================================================================
@torch.no_grad()
def critic(env, model, count=10, device='cpu'):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor(obs)
            mu_v = model(obs_v)[0]
            mu = mu_v.data.cpu().numpy()
            action = np.clip(mu, -1,1)
            obs, r, done, _ = env.step(action)
            rewards += r
            steps += 1
            if done:
                break

    return rewards/count , steps/count


# =============================================================================
# Calculate Log Probabilities
# =============================================================================
def calc_logProb(mu_v, var_v, action_v):
    p1 = - ((action_v - mu_v)**2)/(2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2*pi*var_v))
    return p1+p2


if __name__=='__main__':

    GAMMA = 0.99
    LR = 1e-2
    env = gym.make('MountainCarContinuous-v0')
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    net = A2C(obs_size, act_size)
    agent = A2C_Agent(net)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)



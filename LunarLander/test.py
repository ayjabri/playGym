#!/usr/bin/env python3
'''
Trying to use the exact same code as the one in the book
But the performance is still the same. it doesn't converge
'''
import os
import time
import math
import ptan
import gym
import argparse
from tensorboardX import SummaryWriter

from lib import model, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "LunarLanderContinuous-v2"
GAMMA = 0.99
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
STEPS = 1
TEST_ITERS = 1000
TRAIN_EPISODES = 5
SOLVE_BOUND = 150


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")


    env = gym.make(ENV_ID)
    net = model.ModelRL(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)
    agent = model.AgentRL(net,device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=STEPS)
    optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
    
    s,a,r = [],[],[]
    q_val_refs = []
    episode = 0
    total_rewards = []
    for exp in exp_source:
        s.append(exp.state)
        a.append(exp.action)
        r.append(exp.reward)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            total_rewards.append(new_reward[0])
            mean = np.mean(total_rewards[-100:])
            print(mean)
            if mean > SOLVE_BOUND:
                print('Solved!')
                break
            q_val_refs.extend(common.calc_qval_rl(r,GAMMA))
            r.clear()
            episode += 1
        if episode < TRAIN_EPISODES:
            continue
        
        state_v = torch.FloatTensor(np.array(s, copy=False))
        action_v = torch.FloatTensor(np.array(a, copy=False))
        q_val_refs_v = torch.FloatTensor(np.array(q_val_refs, copy=False))
        s.clear()
        a.clear()
        q_val_refs.clear()
        epispde = 0
        
        optimizer.zero_grad()
        mu_v,var_v = net(state_v)
        log_prob_v = q_val_refs_v.unsqueeze(-1) * calc_logprob(mu_v, var_v, action_v)
        policy_loss = log_prob_v.mean()
        policy_loss.backward()
        optimizer.step()
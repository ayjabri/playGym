# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:27:37 2020

@author: Ayman Al Jabri
It is not advisable to use multiprocessing for a simple environment like CartPole.
The results will be worse than if you ran without it, but this is for demonstration 
purpose only.
"""

import os
import gym
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.multiprocessing as mp

import numpy as np
from time import time
from collections import namedtuple


TotalRewards = namedtuple('TotalRewards', 'reward')

MINI_BATCH_SIZE = 32
STEPS = 4
LR = 1e-3
GAMMA = 0.99
ENV_ID = 'CartPole-v0'
WRAP = False
ENVS = 2
ENTROPY_BETA = 0.02
BATCH_SIZE = 1280
CLIP_GRAD = 0.1
PROCESS_COUNT = 1
REWARD_BOUND = 195
PLAY_EPISODES = 8


def wrap_em(env):
    env = ptan.common.wrappers.NoopResetEnv(env)
    env = ptan.common.wrappers.FireResetEnv(env)
    env = ptan.common.wrappers.EpisodicLifeEnv(env)
    return env 


class A3CNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, 256),
                                  nn.ReLU())
        self.policy = nn.Linear(256, act_size)
        self.val = nn.Linear(256,1)
    
    def forward(self, x):
        base = self.base(x)
        return self.policy(base), self.val(base)
    

def unpack_batch(batch,net,device='cpu'):
    s,a,r,ls = [],[],[],[]
    not_done_idx = []
    for idx, exp in enumerate(batch):
        s.append(np.array(exp.state, copy=False))
        a.append(int(exp.action))
        r.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            ls.append(np.array(exp.last_state, copy=True))
    s_v = torch.FloatTensor(np.array(s, copy=False))
    a_v = torch.LongTensor(a)
    r_np = np.array(r, copy=False)
    
    if not_done_idx:
        ls_v = torch.FloatTensor(np.array(ls, copy=False))
        last_value_np = net(ls_v)[1].data.numpy()[:,0]
        last_value_np *= GAMMA**STEPS
        r_np[not_done_idx] += last_value_np
    ref_value_v = torch.FloatTensor(r_np)
    return s_v, a_v, ref_value_v


def data_func(net,queue,device='cpu'):
    envs = [gym.make(ENV_ID) for _ in range(ENVS)]
    agent = ptan.agent.ActorCriticAgent(net, device=device,apply_softmax=True,
                                        preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA,steps_count=STEPS)
    mini_batch = []
    for exp in exp_source:
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            data = TotalRewards(new_reward[0])
            queue.put(data)
        mini_batch.append(exp)
        if len(mini_batch) < MINI_BATCH_SIZE:
            continue
        data = unpack_batch(mini_batch, net, device)
        queue.put(data)
        mini_batch.clear()

if __name__=="__main__":
    mp.set_start_method('fork', force=True)
    os.environ['OMP_NUM_THREADS']='1'
    env = gym.make(ENV_ID)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    net = A3CNet(obs_size, act_size)
    net.share_memory()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    train_queue = mp.Queue(maxsize=PROCESS_COUNT)
    data_proc_funcs = []
    for _ in range(PROCESS_COUNT):
        data_proc = mp.Process(target=data_func, args=(net,train_queue))
        data_proc.start()
        data_proc_funcs.append(data_proc)
    
    total_rewards = []
    batch_states = []
    batch_actions = []
    batch_ref_q_vals = []
    batch_size = 0
    episodes = 0
    loss = 0.0
    start_time = time()
    try:
        while True:
            train_data = train_queue.get()
            if isinstance(train_data, TotalRewards):
                total_rewards.append(train_data.reward)
                episodes += 1
                mean = np.mean(total_rewards[-100:])
                delta_time = time() - start_time
                if delta_time > 1:
                    print(f'mean: {mean:6.2f}, loss {loss:6.2f}')
                    start_time = time()
                if mean > REWARD_BOUND:
                    print('Solved')
                    break
                continue
            mini_s, mini_a, mini_ref_v = train_data
            
            batch_states.append(mini_s)
            batch_actions.append(mini_a)
            batch_ref_q_vals.append(mini_ref_v)
            batch_size += len(mini_s)
            
            # if episodes < PLAY_EPISODES:
            #     continue
            if batch_size < BATCH_SIZE:
                continue

            states_v = torch.cat(batch_states)
            actions_v = torch.cat(batch_actions)
            ref_val_v = torch.cat(batch_ref_q_vals)
            batch_states.clear()
            batch_actions.clear()
            batch_ref_q_vals.clear()
            batch_size = 0
            episodes = 0
          
            optimizer.zero_grad()
            policy, val = net(states_v)
            value_loss = F.mse_loss(val.squeeze(-1), ref_val_v)
            
            log_prob = F.log_softmax(policy, dim=1)
            log_prob_a = log_prob[range(len(actions_v)),actions_v]
            adv_v = ref_val_v - val.detach().squeeze(-1)
            policy_loss = - (log_prob_a * adv_v).mean()
            
            prob = F.softmax(policy, dim=1)
            ent = (prob * log_prob).sum(dim=1).mean()
            entropy_loss = ENTROPY_BETA * ent
            
            loss = value_loss + policy_loss + entropy_loss
            loss.backward()
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
    finally:
        for p in data_proc_funcs:
            p.terminate()
            p.join()
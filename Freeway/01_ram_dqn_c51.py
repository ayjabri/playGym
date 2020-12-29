#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 08:51:18 2020

@author: ayman
"""

import gym
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from time import time
from datetime import datetime
from lib import model, utils
import argparse


ENV_ID = 'Freeway-ram-v0'
# ENV_ID = 'Pong-ram-v0'
# ENV_ID = 'CartPole-v0'
GAMMA = 0.99
Vmin = 0
Vmax = 20
N_ATOMS = 51
support = np.linspace(Vmin, Vmax, N_ATOMS)
STEPS = 10
BUFFER_SIZE = 50_000
SOLVE = 20
INIT_REPLAY = 10_000
BATCH_SIZE = 64
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.02
EPS_STR_FRAME = 10_000
EPS_END_FRAME = 50_000
N_ENVS = 4
SEED = 120


def play(env, agent, render=True):
    state = env.reset()
    rewards = 0
    while True:
        if render: env.render()
        state_v = ptan.agent.float32_preprocessor(state).unsqueeze(0)
        action,_ = agent(state_v)
        new_state, r,d,_ = env.step(action[0])
        rewards += r
        if d:
            print(rewards)
            break
        state = new_state
    env.close()


class DistDQN(nn.Module):
    def __init__(self, obs_size, act_size, Vmin, Vmax, NATOMS):
        super().__init__()
        self.atoms = NATOMS
        self.vmin = Vmin
        self.vmax = Vmax
        self.layer = nn.Sequential(nn.Linear(obs_size, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, act_size * NATOMS)
                                   )
        self.register_buffer('support', torch.linspace(Vmin, Vmax, NATOMS))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        return self.layer(x).view(batch_size, -1, self.atoms)
    
    def apply_softmax(self, output):
        output_size = output.shape[0]
        return self.softmax(output.view(-1, self.atoms)
                            ).view(output_size,-1,self.atoms)
    
    def both(self, x):
        out = self(x)
        probs = self.apply_softmax(out)
        weights = probs * self.support
        res = weights.sum(dim=2)
        return out, res
    
    def qval(self,x):
        return self.both(x)[1]


def distr_projection(next_distr, rewards, dones, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS),
                          dtype=np.float32)
    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    for atom in range(N_ATOMS):
        v = rewards + (Vmin + atom * delta_z) * gamma
        tz_j = np.minimum(Vmax, np.maximum(Vmin, v))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += \
            next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += \
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += \
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(
            Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = \
                (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = \
                (b_j - l)[ne_mask]
    return proj_distr


def update_dist(r, support, probs_action, vmin, vmax,gamma):
    '''
    Takes numpy arrays
    r: one reward at a time
    support: range of natoms
    probs_actions: the probablity distribution of next actions from target network 
    '''
    natoms = probs_action.shape[0]
    dz = (vmax-vmin)/(natoms-1)
    bj = np.round((r-vmin)/dz)
    bj = (np.clip(bj, 0, natoms-1)).astype(np.int)
    m = np.copy(probs_action)
    j = 1
    for i in range(bj, 1, -1):
        m[i] += np.power(gamma,j)*m[i-1]
        j += 1
    j = 1
    for i in range(bj, natoms-1, 1):
        m[i] += np.power(gamma, j) * m[i+1]
        j += 1
    m /= m.sum()
    return m


def update_batch_dist(probs_actions, rewards, done, support, vmin, vmax, gamma):
    batch_size,natoms = probs_actions.shape
    o = np.zeros((batch_size,natoms))
    dz = (vmax-vmin)/(natoms-1)
    for p in range(batch_size):
        if not done[p]:
            o[p] = update_dist(rewards[p], support, probs_actions[p], vmin, vmax, gamma)
        else:
            bj = np.round((rewards[p]-vmin)/dz)
            bj = np.clip(bj, 0,natoms-1).astype(np.int)
            o[p,bj]=1.0
    return o


def unpack_dqn(batch):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    return (np.array(states, copy=False), 
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(last_states, copy=False))


def calc_dqn_loss(batch, net, tgt_net, gamma=GAMMA):
    states, actions, rewards, dones, last_states = unpack_dqn(batch)
    states_v = torch.FloatTensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    dones_v = torch.BoolTensor(dones)
    last_states_v = torch.FloatTensor(last_states)
    
    state_action_value = net.qval(states_v)[range(len(actions_v)), actions_v]
    with torch.no_grad():
        target_state_action = tgt_net.target_model.qval(last_states_v).max(dim=1)[0]
        target_state_action[dones_v] = 0.0
    q_state_action = rewards_v + target_state_action * gamma
    return F.mse_loss(state_action_value, q_state_action)
    

def calc_dist_loss(batch, net, tgt_net, gamma=GAMMA):
    states, actions, rewards, dones, last_states = unpack_dqn(batch)
    
    states_v = torch.FloatTensor(states)
    actions_v = torch.tensor(actions)
    last_states_v = torch.FloatTensor(last_states)
    
    next_dist, next_qvals = tgt_net.target_model.both(last_states_v)
    next_acts = next_qvals.max(dim=1)[1].data.numpy() #next actions
    next_probs = tgt_net.target_model.apply_softmax(next_dist).data.numpy()
    next_probs_actions = next_probs[range(len(next_acts)), next_acts]
    
    proj_dist = update_batch_dist(next_probs_actions, rewards, dones, support, Vmin, Vmax, GAMMA)
    # proj_dist = distr_projection(next_probs_actions,rewards,dones, GAMMA)
    proj_dist_v = torch.FloatTensor(proj_dist)
    
    distr_v = net(states_v)
    sa_values = distr_v[range(len(actions_v)), actions_v.data]
    log_sa_values = F.log_softmax(sa_values, dim=1)
    loss = -log_sa_values * proj_dist_v
    return loss.sum(dim=1).mean()



def calc_loss_(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_dqn(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_acts = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v)
    next_distr = next_distr.data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_acts]

    proj_distr = distr_projection(
        next_best_distr, rewards, dones, GAMMA)


    distr_v = net(states_v)
    sa_vals = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(sa_vals, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default= 'freeway', help='select an atari game to play: cartpole, freeway or beamrider')
    parser.add_argument('--nenvs', default=1, type=int, help='select the number of environment to extract sample transitions from')
    parser.add_argument('--steps', default=1, type=int, help='number of steps to skip when training')
    parser.add_argument('--wraplife', default=True, type=bool, help='Wrap games that requires episode end')
    args = parser.parse_args()
    
    params = model.HYPERPARAMS[args.env]
    def make_env(ENV_ID):
        return ptan.common.wrappers.EpisodicLifeEnv(gym.make(ENV_ID))
    envs = []
    for _ in range(N_ENVS):
        if args.wraplife:
            env = make_env(ENV_ID)
        else:
            env = gym.make(ENV_ID)
        env.seed(SEED)
        envs.append(env)
        
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    
    net = model.DistDQN(obs_size, act_size,Vmin, Vmax, N_ATOMS)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()

    epsilon_tracker = model.EpsilonTracker(selector, eps_start=EPS_START,\
                                           eps_final=EPS_END,
                                           eps_start_frame = EPS_STR_FRAME,
                                           eps_end_frame = EPS_END_FRAME)
    agent = ptan.agent.DQNAgent(lambda x: net.qval(x), selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, BUFFER_SIZE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    total_rewards = deque(maxlen=100)
    frame_idx = 0
    episode = 0
    st = datetime.now()
    pt = time()
    loss =0.0
    
    while True:
        frame_idx += 1
        epsilon_tracker.frame(frame_idx * STEPS)
        buffer.populate(STEPS)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            episode += 1
            total_rewards.append(new_reward[0])
            mean = np.mean(total_rewards)
            if time()-pt > 2:
                print(f'{frame_idx*STEPS:7,}: episode:{episode:4}, mean:{mean:7.2f}, loss:{loss:7.2f}, epsilon:{selector.epsilon:6.2f}')
                pt = time()
            if mean > SOLVE:
                print(f'Solved within {datetime.now()-st}')
                break
        if len(buffer) < INIT_REPLAY*N_ENVS:
            continue

        batch = buffer.sample(BATCH_SIZE * N_ENVS)
        optimizer.zero_grad()
        # loss = calc_dqn_loss(batch, net, tgt_net, GAMMA)
        loss = calc_loss_(batch, net, tgt_net.target_model, GAMMA)
        # loss = calc_dist_loss(batch, net,tgt_net,GAMMA)
        loss.backward()
        optimizer.step()
        
        if frame_idx % 1000 ==0:
            tgt_net.sync()
        
            
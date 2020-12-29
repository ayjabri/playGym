#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:03:29 2020

@author: aa97842
"""

import gym
import ptan
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta

class Net(nn.Module):
    def __init__(self, observation_shape, n_actions):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(observation_shape[0], 128),
                                 nn.ReLU(),
                                 nn.Linear(128, n_actions)
                                 )
        
    def forward(self, x):
        return self.lin(x.float())



class ActorCritic(nn.Module):
    def __init__(self, shape, action_shape):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(shape[0], 256),
                                    nn.ReLU()
                                    )
        
        self.value = nn.Linear(256, 1)
                                   
        self.mu = nn.Sequential(nn.Linear(256, action_shape),
                                nn.Tanh())
        self.var = nn.Sequential(nn.Linear(256, action_shape),
                                 nn.Softplus())

    def forward(self, obs):
        state = self.input(obs)
        return self.mu(state), self.var(state), self.value(state)


@torch.no_grad()
def play_episode(env, net):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        action = net(torch.FloatTensor(state)).argmax(dim=-1).item()
        last_state, reward, done,_= env.step(action)
        rewards += reward
        if done:
            print('Rewards: {}'.format(rewards))
            break
        state = last_state
    env.close()


def unpack_batch(batch):
    states =[]
    actions = []
    rewards = []
    last_states = []
    dones = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    return (np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(last_states))


def calc_loss(batch, net, tgt_net, gamma):
    states, actions, rewards, dones, last_states = unpack_batch(batch)
    
    states_v = torch.FloatTensor(states)
    actions_v = torch.LongTensor(actions)
    rewards_v = torch.tensor(rewards)
    # done_tags = torch.BoolTensor(dones)
    last_states_v = torch.FloatTensor(last_states)
    
    q_s = net(states_v)
    q_sa = q_s.gather(dim=-1, index=actions_v.view(-1,1))
    with torch.no_grad():
        max_q_sa_prime = tgt_net.target_model(last_states_v).max(dim=-1)[0]
        max_q_sa_prime[dones] = 0.0
        target_q_sa = max_q_sa_prime * gamma + rewards_v
    return F.mse_loss(target_q_sa, q_sa.squeeze(-1))

if __name__=='__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--play', action='store_true', help='Play an episode after training is complete')
    parser.add_argument('--save',action='store_true', default=False, help='Store a copy of the network')
    args = parser.parse_args()

    gamma = 0.95
    steps = 2
    env = gym.make('CartPole-v0')
    net = Net(env.observation_space.shape, env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)
    
    selector=ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, 1.0, 0.05, 20_000)
    agent = ptan.agent.DQNAgent(net, selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=steps)
    gamma = gamma**steps
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, 100_000)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
    
    
    frame_idx=0
    episode = 0
    rewards = []
    start_time = datetime.now()
    while True:
        frame_idx +=1
        eps_tracker.frame(frame_idx)
        
        buffer.populate(1)
        
        reward = exp_source.pop_total_rewards()
        if reward:
            episode += 1
            rewards.append(reward)
            mean = np.mean(rewards[-100:])
            print(f'Episde:{episode} Mean:{mean:0.2f}, Epsilon:{selector.epsilon:0.2f}')
            if mean > 195:
                print("Solved in {}".format(timedelta(seconds=(datetime.now()-start_time).seconds)))
                break
        
        if len(buffer) < 10_000:
            continue
        
        optimizer.zero_grad()
        batch = buffer.sample(200)
        loss = calc_loss(batch, net, tgt_net, gamma)
        loss.backward()
        optimizer.step()
        
        if frame_idx % 1000 ==0:
            tgt_net.sync()
    
    if args.save: torch.save(net, 'trainedModels/actorCritic')
    if args.play: play_episode(env, net)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:27:52 2020

@author: Ayman Al Jabri

This is an imporovement to the Reinforce method,
but nowadays, almost nobody uses plain-vanilla policy gradient method, for the much
more stable actor-critic method. However, I still want to experement PG implementation.
It establishes very important concepts and metrics that will come handy when coding
Actor critic.

Solving Lunar lander using Policy Gradient method:
    Network: Simple 2 layers with one output returning mu
    Observations: fresh episodes from ptan experience-source (must complete full episode to start training)
    Rewards: discounted using ptan STEPS = 10 then scale using frame_idx 
    Loss is the sum of:
        1- Policy Loss:is the negative mean log of probability (log_soft), multiplied by discounted rewards
        2- Entropy Loss: the sum of probabilities times log_prob, then take the negative average of that
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import ptan
import argparse
from datetime import datetime, timedelta
from time import time


# =============================================================================
# Simple NN with two layers
# =============================================================================
class Net(nn.Module):
    def __init__(self, shape, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(shape, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, n_actions)
                                   )
        
    def forward(self, x):
        return self.layer(x)

# =============================================================================
# Scale losses using baseline discount
# =============================================================================
def calc_scale_batch(batch_rewards,baseline=True):
    if not baseline: return batch_rewards
    baseline = np.cumsum(batch_rewards)/np.arange(1,len(batch_rewards)+1)
    return batch_rewards - baseline

# =============================================================================
# Play function to execute when training is complete
# =============================================================================
@torch.no_grad()
def play(env,agent):
    state = env.reset()
    rewards= 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()  
   
# =============================================================================
# Hyperparameters
# =============================================================================
GAMMA = 0.99
LR = 1e-3
SOLVE = 150
ENTROPY_BETA = 0.01

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='Play and episode when training is complete')
    parser.add_argument('--save', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_pg.dat"')
    parser.add_argument('--episodes', '-e', default=10, type=int, help='Episodes to train on in each epoch')
    parser.add_argument('--steps','-s', default=10, type=int, help='Steps used to discount reward') # This is new in Policy Gradient Method
    parser.add_argument('--baseline','-b', action='store_true', default=True, help='Steps used to discount reward') # This is new in Policy Gradient Method
    args = parser.parse_args()
    env = gym.make('LunarLander-v2')
    net = Net(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=args.steps)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    batch_s, batch_a, batch_r, batch_qval = [],[],[],[]
    total_rewards =[]
    episode = 0
    epoch = 0

    print(net)
    start_time = datetime.now()
    print_time = time()
    for idx,exp in enumerate(exp_source):
        batch_s.append(exp.state)
        batch_a.append(exp.action)
        batch_r.append(exp.reward)

        reward = exp_source.pop_total_rewards()
        if reward:
            batch_qval.extend(calc_scale_batch(batch_r, args.baseline))
            batch_r.clear()
            episode += 1
            total_rewards.append(reward[0])
            mean = np.mean(total_rewards[-100:])
            if time()- print_time >1:
                print(f'epoch:{epoch:6} mean:{mean:7.2f}, loss:{loss:7.2f}, reward{reward[0]:7.2f}')
                print_time = time()
            if mean > SOLVE:
                duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                print('Solved in {duration}')
                fname = ('lunar_pg_baseline.dat' if args.baseline else 'lunar_pg.dat')
                if args.save: torch.save(net.state_dict(),fname)
                if args.play: play(env,agent)
                break
        
        if episode < args.episodes:
            continue
        epoch += 1
        
        state_v = torch.FloatTensor(batch_s)
        act_v = torch.LongTensor(batch_a)
        qval_v = torch.FloatTensor(batch_qval)

        optimizer.zero_grad()
        logit_v = net(state_v)
        log_prob_v = F.log_softmax(logit_v, dim=1)
        log_prob_a_v = log_prob_v[range(len(act_v)),act_v]
        policy_loss = - (log_prob_a_v * qval_v).mean()
        
        prob_v = F.softmax(logit_v, dim=1)
        entropy = (prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss =  ENTROPY_BETA * entropy
        
        loss = policy_loss + entropy_loss
        loss.backward()
        optimizer.step()
        
        batch_s.clear()
        batch_a.clear()
        batch_r.clear()
        batch_qval.clear()
        episode = 0





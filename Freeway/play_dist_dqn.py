#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:25:22 2020

@author: Ayman Al Jabri
"""
import ptan
import gym
from lib import model
import argparse
import torch


@torch.no_grad()
def play(env, net):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        state_v = ptan.agent.float32_preprocessor([state])
        qvals = net.qval(state_v)
        action = qvals.argmax().item()
        new_state,r,d,_ = env.step(action)
        rewards += r
        if d:
            print(rewards)
            break
        state = new_state
    env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='freeway', help='Choose an envrironment to play')
    args = parser.parse_args()
    param = model.HYPERPARAMS[args.env]
    env = gym.make(param.ENV_ID)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    net = model.DistDQN(obs_size, act_size, param.Vmin,param.Vmax,param.N_ATOMS)
    fname = param.ENV_ID + '_best_dist.dat'
    net.load_state_dict(torch.load(fname))
    play(env,net)
    
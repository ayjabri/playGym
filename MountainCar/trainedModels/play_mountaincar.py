# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:20:27 2020

@author: ayjab
"""
import time
import gym
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F




class DuelDQN(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()

        self.input = nn.Linear(in_features, 256)
        self.fc_adv = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.input(x.float()))
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return (val + (adv - adv.mean(dim=-1, keepdim=True)))



@torch.no_grad()
def play(env, net):
    state = env.reset()
    rewards = 0.0
    while True:
        env.render()
        time.sleep(0.01)
        stateV = torch.FloatTensor(state)
        action = net(stateV).argmax(dim=-1).item()
        next_state,reward,done,_= env.step(action)
        rewards += reward
        if done:
            print(rewards)
            break
        state = np.array(next_state)
    time.sleep(1)
    env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fname', type=str, default='MountainCar_False_1_perfect.dat', help='Enter thel filename of the DuelNN state dictionary')
    args = parser.parse_args()

    env = gym.make('MountainCar-v0')

    in_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net=DuelDQN(in_features, n_actions)
    net.load_state_dict(torch.load(args.fname))

    play(env, net)



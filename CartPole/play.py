# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:34:38 2020

@author: ayjab
"""
import gym
import torch
import torch.nn as nn
import argparse


class REL(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions))

    def forward(self, x):
        return self.layer(x)
    
    

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
    
 
net = REL(4,2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', required=True, default='trainedModels/reinforce', type=str ,help='path to trained model of CartPole v0')
    parser.add_argument('-n', default=1, type=int, required=False, help='Number of times to play')
    args = parser.parse_args()
    env = gym.make('CartPole-v0')
    net = torch.load(args.path)
    for i in range(args.n):
        play_episode(env, net)
    
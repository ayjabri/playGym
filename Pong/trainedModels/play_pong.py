# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:32:09 2020

@author: ayjab
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time
import numpy as np
import ptan



class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            b = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(b)
            self.register_buffer('epsilon_bias', torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.data.normal_()
        w = self.weight + self.sigma_weight * self.epsilon_weight.data
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.data.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, w, bias)



class DuelNoisyDQN(nn.Module):
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]

        self.fc_adv = nn.Sequential(NoisyLinear(outshape, 256),
                                nn.ReLU(),
                                NoisyLinear(256, n_actions))
        self.fc_val = nn.Sequential(NoisyLinear(outshape, 256),
                                nn.ReLU(),
                                NoisyLinear(256, 1))

    def forward(self, x):
        x = self.conv(x.float() / 256)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return (val + (adv - adv.mean(dim=1, keepdim=True)))



env_name = 'PongNoFrameskip-v4'
env = gym.make(env_name)
env = ptan.common.wrappers.wrap_dqn(env, stack_frames=4)

net = DuelNoisyDQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load('pong_1_4_.dat', map_location='cpu'))
net.eval()

@torch.no_grad()
def play(env, net=None):
    state = np.array(env.reset())
    rewards = 0.0
    while True:
        env.render()
        time.sleep(0.02)
        if net is not None:
            stateV = torch.FloatTensor([state])
            action = net(stateV).argmax(dim=-1).item()
        else:
            action = env.action_space.sample()
        next_state,reward,done,_= env.step(action)
        rewards += reward
        if done:
            print(rewards)
            break
        state = np.array(next_state)
    time.sleep(0.5)
    env.close()

if __name__=='__main__':
    play(env, net)
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:21:21 2020

@author: aa97842
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, product, permutations


class DQN(nn.Module):
    def __init__(self,img_height):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=img_height,out_channels=32 , kernel_size = 7, stride = 2)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels=32 , kernel_size = 7, stride = 2)
        self.conv3 = nn.Conv1d(in_channels=32,out_channels=16 , kernel_size = 7, stride = 2)
        self.adapt = nn.AdaptiveMaxPool1d(1)
        self.flat  = nn.Flatten()
        self.out   = nn.Linear(16, 2)
    
    def forward(self,s):
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = self.flat(self.adapt(s))
        s = self.out(s)
        return s


Experience = namedtuple('Experience',('state','action','next_state','reward'))

class ReplayMemory():
    def __init__(self,capacity):
        self.capacity = capacity
        self.mem = deque(maxlen=self.capacity)
        self.count = 0
        
    def store(self,experience):
        self.mem.append(experience)
        self.count += 1
    def __len__(self):
        return len(self.mem)
    
    def sample(self,batch_size):
        batch_size = min(batch_size,len(self.mem))
        return random.sample(self.mem, batch_size)

class EpsilonGreedy(object):
    def __init__(self,start,end,decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def exploration_rate(self,current_step):
        rate = self.end + (self.start - self.end) * np.exp(-self.decay * current_step)
        return rate


class Agent(object):
    def __init__(self,strategy):
        self.strategy = strategy
        self.current_step = 0
        self.explore = True
        
    def next_action(self,policy_net,state,env):
        if random.random() > self.strategy.exploration_rate:
            action = policy_net(state).argmax(dim=1)
            self.explore = False
        else:
            action = torch.tensor([env.env.action_space.sample()])
            self.explore = True
        return action
    
    
class PlotIt():
    def __init__(self,values,period):
        self.values = values
        self.period = period
        self.fig,self.ax = plt.subplots(num =1)
    def plot(self):
        move_avg = self.values.unfold(1,self.period,1).mean(1)
        zeros = torch.zeros(self.period)
        move_avg = torch.cat((zeros,move_avg))
        plt.title('Training ....')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.plot(self.values)
        plt.plot(move_avg)
        


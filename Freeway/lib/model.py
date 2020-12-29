#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:26:36 2020

@author: ayman
"""
import torch
import torch.nn as nn
import ptan
from typing import Union
from types import SimpleNamespace
import numpy as np

HYPERPARAMS = ({'cartpole':
          SimpleNamespace(**{'ENV_ID':'CartPole-v0',
           'Vmin': 0,
           'GAMMA': 0.99,
           'Vmax': 200,
           'N_ATOMS': 51,
           'STEPS': 10,
           'BUFFER_SIZE': 20_000,
           'SOLVE': 195,
           'INIT_REPLAY': 1000,
           'BATCH_SIZE' : 64,
           'LR' : 1e-3,
           'EPS_START' : 1.0,
           'EPS_END' : 0.02,
           'EPS_FRAME' : 5000,
           'EPS_START_FRAME' : 1000,
           'EPS_END_FRAME' : 5000,
           'SEED' : 124
           }),
          'freeway':
          SimpleNamespace(**{'ENV_ID':'Freeway-ram-v0',
           'Vmin': 0,
           'GAMMA': 0.99,
           'Vmax': 22,
           'N_ATOMS': 51,
           'STEPS': 10,
           'BUFFER_SIZE': 50_000,
           'SOLVE': 20,
           'INIT_REPLAY': 10_000,
           'BATCH_SIZE' : 64,
           'LR' : 4e-4,
           'EPS_START' : 1.0,
           'EPS_END' : 0.02,
           'EPS_FRAME' : 25_000,
           'EPS_START_FRAME' : 10_000,
           'EPS_END_FRAME' : 50_000,           
           'SEED' : 124
           }),
          'pong':
          SimpleNamespace(**{'ENV_ID':'Pong-ram-v0',
           'Vmin': -21,
           'GAMMA': 0.99,
           'Vmax': 21,
           'N_ATOMS': 51,
           'STEPS': 10,
           'BUFFER_SIZE': 100_000,
           'SOLVE': 20,
           'INIT_REPLAY': 10_000,
           'BATCH_SIZE' : 64,
           'LR' : 4e-4,
           'EPS_START' : 1.0,
           'EPS_END' : 0.02,
           'EPS_FRAME' : 25_000,
           'EPS_START_FRAME' : 1,
           'EPS_END_FRAME' : 25_000,           
           'SEED' : 124
           }),
          })
    

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
    
    
class DQNNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(obs_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, act_size),
                                   )
    def forward(self, x):
        return self.layer(x)


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(obs_size, 400),
                                   nn.ReLU(),
                                   nn.Linear(400, 300),
                                   nn.ReLU(),
                                   nn.Linear(300, act_size),
                                   )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.layer(x)
    
    def apply_softmax(self,out):
        return self.softmax(out)
    
    
class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.net_in = nn.Sequential(nn.Linear(obs_size, 400),
                                  nn.ReLU(),
                                  )
        self.net_out= nn.Sequential(nn.Linear(400 + 1, 300),
                                    nn.ReLU(),
                                    nn.Linear(300, 1)
                                    )
    def forward(self, s, a):
        out = self.net_in(s)
        return self.net_out(torch.cat([out,a],dim=1))
    

class EpsilonTracker(ptan.actions.EpsilonTracker):
    """
    Custom Epsilon Greedy Tracker that starts reducing epsilon at certain frame
    Useful for environments that require more exploration at initiation
    """
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_start_frame: int,
                 eps_end_frame: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_start_frame = eps_start_frame
        self.eps_end_frame = eps_end_frame
        self.selector.epsilon = 1.0

    def frame(self, frame: int):
        if frame > self.eps_start_frame:
            eps = self.eps_start - (frame - self.eps_start_frame) / \
                (self.eps_end_frame - self.eps_start_frame)
            self.selector.epsilon = max(self.eps_final, eps)
        pass


class DDPGAgent(ptan.agent.BaseAgent):
    def __init__(self, model, selector=ptan.actions.ArgmaxActionSelector(),
                 preprocessor=None, apply_softmax=True, epsilon=0.3, device='cpu'):
        self.model=model
        if preprocessor:
            self.preprocessor=preprocessor
        else:
            self.preprocessor=ptan.agent.float32_preprocessor
        self.epsilon = epsilon
        self.apply_softmax = apply_softmax
        self.device = device
        self.selector = selector
    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        states_v = self.preprocessor(states).to(self.device)
        output = self.model(states_v)
        output += torch.randn(output.shape) * self.epsilon
        if self.apply_softmax:
            output = torch.softmax(output,dim=1)
        actions = self.selector(output.data.numpy())
        return actions, agent_states
    
        
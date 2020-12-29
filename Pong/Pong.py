#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 05:53:43 2020

@author: aymanjabri
"""

import random
import torch
import torch.nn as nn
import argparse
import numpy as np
import wrappers as wrap
import torch.nn.functional as F
from collections import namedtuple, deque



# DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
DEFAULT_ENV_NAME = "Pong-v0"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = namedtuple('Experience',('state','action','reward', 'done', 'next_state'))

class DQN(nn.Module):
    def __init__(self,frame_shape,num_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(frame_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        conv_output = self.conv(torch.zeros(1,*frame_shape)).shape[1]
                                
        self.fc = nn.Sequential(nn.Linear(conv_output, 512),
                                nn.ReLU(),
                                nn.Linear(512, num_actions)
                                )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer.clear()
    
    def store(self, experience):
        self.buffer.append(experience)
    
    def sample(self, num):
        size = min(num, len(self.buffer))
        sample = random.sample(self.buffer, size)
        s,a,r,d,n = zip(*sample)
        return np.array(s),np.array(a),np.array(r),np.array(d),np.array(n)
    
    def can_sample(self, MIN_BATCH_SIZE):
        return len(self.buffer) > MIN_BATCH_SIZE

    
class Agent():
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self.reset()
        
    def reset(self):
        self.state = self.env.reset()
        self.total_rewards = 0.0
    
    @torch.no_grad()
    def play_episode(self, net, epsilon=0.0, device='cpu'):
        done_rewards = None
        state = self.state
        
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.tensor([state])
            q_val = net(state_v)
            _,act_v = q_val.max(dim=1)
            action = int(act_v.item())
            
        next_state, reward, done, _ = self.env.step(action)
        self.total_rewards += reward
        
        e = Experience(state,action,reward,done,next_state)
        self.buffer.store(e)
        self.state = next_state
        if done:
            done_rewards = self.total_rewards
            self.reset()
        return done_rewards
    
def calc_loss(batch, net, tgt_net, device='cpu'):
    s,a,r,d,n = batch
    states = torch.tensor(s)
    actions = torch.tensor(a, dtype=torch.int64)
    rewards = torch.tensor(r)
    done_flags = torch.tensor(d, dtype=torch.bool)
    new_states = torch.tensor(n)
    
    state_action_value = net(states).gather(dim=1,index=actions.view(-1,1)).squeeze(-1)
    next_state_value = tgt_net(new_states).max(dim=1)[0]
    next_state_value[done_flags] = 0.0
    next_state_value.detach_()
    
    expected_state_value = rewards + GAMMA * next_state_value
    return F.mse_loss(expected_state_value, state_action_value)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true',help='Enable Cuda')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME, help='Enter environment, default: ' + \
                        DEFAULT_ENV_NAME)
    parser.add_argument('--reward', type=float, default=MEAN_REWARD_BOUND, help='Reward Bound')
    arg = parser.parse_args()
    
    device = ('cuda' if arg.cuda else 'cpu')
    env = wrap.make_env(arg.env)
    
    frame_shape = env.observation_space.shape
    num_actions = env.action_space.n
    net = DQN(frame_shape,num_actions).to(device)
    tgt_net = DQN(frame_shape,num_actions).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    buffer = ReplayMemory(REPLAY_SIZE)
    agent = Agent(env, buffer)
    optimizer = torch.optim.Adam(net.parameters(),lr = LEARNING_RATE)
    
    total_rewards = []
    frame_idx = 0
    
    while True:
        frame_idx += 1
        eps = max(EPSILON_FINAL, EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
        
        reward = agent.play_episode(net, epsilon=eps, device=device)
        if reward is not None:
            total_rewards.append(reward)
            mean_rewards = np.mean(total_rewards[-100:])
            print(f'{frame_idx}: Episode:{len(total_rewards):.0f}, EPS:{eps:.3f},mean:{mean_rewards:.3f}')
            if mean_rewards >= MEAN_REWARD_BOUND:
                print('Solved!')
                break
            
        if len(buffer) < REPLAY_SIZE:
            continue
        
        if frame_idx % SYNC_TARGET_FRAMES == 0 :
            tgt_net.load_state_dict(net.state_dict())
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, net, tgt_net, device=device)
        loss.backward()
        optimizer.step()
        
    
if __name__=='__main__':
    main()
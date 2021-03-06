# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:32:09 2020

@author: ayjab
"""

from lib import fast_wrappers, my_models
import torch
import time
import numpy as np



env_name = 'BreakoutNoFrameskip-v4'
env = fast_wrappers.make_atari(env_name)
env = fast_wrappers.wrap_deepmind(env, clip_rewards=False, frame_stack=True, frame_stack_count=2)

net = my_models.DuelDQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load('breakout-small_1_2_.dat', map_location='cpu'))
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
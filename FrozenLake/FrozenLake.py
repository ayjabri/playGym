#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:37:29 2020

@author: aymanjabri
"""

import gym
import time
import random
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from IPython.display import clear_output


######## Create the envisornment and set up its variables
# env = gym.make('FrozenLake-v0')
# # env = gym.make('FrozenLake8x8-v0')
# number_of_actions = env.action_space.n
# number_of_states = env.observation_space.n

# ####### Build the Q table that we'll be using to reference actions
# q_table = np.zeros((number_of_states,number_of_actions))

# ####### Set the parameters
# episodes = 10_000
# max_steps = 100

# lr = 0.1
# gamma = 0.99

# exploration_rate = 1
# max_exploration_rate = 1
# min_exploration_rate = 0.01
# exploration_decay_rate = 0.001


# ##### Learning Loop
# rewards_all_episodes = []

def fit(lr,gamma):
    env = gym.make('FrozenLake-v0')
    # env = gym.make('FrozenLake8x8-v0')
    number_of_actions = env.action_space.n
    number_of_states = env.observation_space.n
    q_table = np.zeros((number_of_states,number_of_actions))
    episodes = 10_000
    max_steps = 100
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001
    rewards_all_episodes = []
    
    for episode in range(episodes):
        rewards_episode = 0
        done = False
        state = env.reset() 
        ## loop over all steps in one episode
        for step in range(max_steps):
            '''Available Actions:
                Left  = 0 
                Down  = 1
                Right = 2
                Up    = 3
                '''
            # Exploration or exploitation trade off
            
            e_rate_threshold = random.uniform(0,1)
            if e_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:])
            else:
                action = env.action_space.sample()
            
            
            ## Take the action and record reward and the new state
            new_state,reward,done,info = env.step(action)
            
            # Update the Q table with the new values of Q in (s,a)
            # q_table[state, action] = q_table[state, action] * (1 - lr) + \
            #     lr * (reward + gamma * np.max(q_table[new_state, :]))
            
            q_table[state,action] = q_table[state,action] + lr * \
                (reward + gamma * np.max(q_table[new_state,:]) - q_table[state,action])
            
            state = new_state
            rewards_episode += reward
            
            if done: break
        
        rewards_all_episodes.append(rewards_episode)    
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    results = {}
    rewards_per_thousand = np.split(np.array(rewards_all_episodes),episodes/1000)
    print('\n\n','*'*10,' Rewards Per 1000 Episode ','*'*10,'\n\n')
    for i,j in enumerate(rewards_per_thousand):
        results[i] = j.mean()
        print(i,j.mean())
    return results,q_table


def optimize(lr_range,gamma_range,plot = True):
    train = product(lr_range,gamma_range)
    d = {}
    for lr,gamma in train:
        d[(lr,gamma)],_ = fit(lr,gamma)
    
    if plot:
        plt.figure(figsize=(10,8))
        for h in d.keys():
            plt.plot(list(d[h].values()),label = h)
            plt.legend()

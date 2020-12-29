#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:27:51 2020

@author: aymanjabri
"""

import gym
import time
from collections import defaultdict
from itertools import count

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20



class Agent():
    def __init__(self, ENV_NAME):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = defaultdict(float)
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n
        
    def sample_env(self):
        old_state = self.state
        action = self.env.action_space.sample()
        state, reward, done,_ = self.env.step(action)
        self.state = self.env.reset() if done else state
        return (old_state, action, reward, state)
    
    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.n_actions):
            val = self.values[(state, action)]
            if best_action is None or val > best_value:
                best_value = val
                best_action = action
        return best_value, best_action
    
    def update_values(self, s, a, r, next_s):
        best_value,_ = self.best_value_and_action(next_s)
        old_value = self.values[(s, a)]
        new_value = r + GAMMA * best_value
        self.values[(s, a)] = (1- ALPHA) * old_value + ALPHA * new_value
        
    def play_episode(self, env):
        state  = env.reset()
        total_reward = 0.0
        while True:
            _,action = self.best_value_and_action(state)
            new_state, R, done, _ = env.step(action)
            total_reward += R
            if done:
                break
            state = new_state
        return total_reward
 
    
    
if __name__=="__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME)
    startTime = time.process_time()
    best_reward = 0.0
    
    for step in count():
        agent.update_values(*(agent.sample_env()))

        rewards = 0.0
        for episode in range(TEST_EPISODES):
            rewards += agent.play_episode(test_env)
        rewards /= TEST_EPISODES
        if rewards > best_reward:
            best_reward = rewards    
            print(f'Episode:{step}, Accuracy: {rewards}')
        if rewards >= 0.8:
            break
        
            
    
    
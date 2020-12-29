#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:24:27 2020

@author: aymanjabri
"""

import gym
import time
from collections import defaultdict, Counter
from itertools import count


ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9


class Agent():
    def __init__(self, ENV_NAME):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n
        self.values = defaultdict(float)
        self.transit = defaultdict(Counter)
        self.rewards = defaultdict(float)
        
    def play_random_n(self,n):
        for episode in range(n):
            action = self.env.action_space.sample()
            new_state,r,done,_=self.env.step(action)
            self.transit[(self.state, action)][new_state] += 1
            self.rewards[(self.state, action, new_state)] = r
            self.state = self.env.reset() if done else new_state
            
    def calc_action_value(self, state, action):
        action_value = 0.0
        tgt_counts = self.transit[(state, action)]
        total = sum(tgt_counts.values())
        for target_state, num in tgt_counts.items():
            reward = self.rewards[(state, action, target_state)]
            action_value += num/total * (reward + GAMMA * self.values[target_state])
        return action_value
    
    def best_action(self, state):
        action_value, best_action = None, None
        for action in range(self.n_actions):
            val = self.calc_action_value(state, action)
            if best_action is None or val > action_value:
                action_value = val
                best_action = action
        return best_action
    
    def play_episode(self):
        state = self.env.reset()
        rewards = 0.0
        while True:
            action = self.best_action(state)
            new_state,R,done,_=self.env.step(action)
            rewards += R
            if done:
                break
            state = new_state
        return rewards
    
    def update_values(self):
        for state in range(self.n_states):
            actions_values = [self.calc_action_value(state, action) \
                              for action in range(self.n_actions)]
            self.values[state] = max(actions_values)


if __name__ == '__main__':
    agent = Agent(ENV_NAME)
    n = 1_000
    start_time = time.process_time()
    for episode in count():
        agent.play_random_n(n)
        agent.update_values()
        episode_rewards = 0.0
        for _ in range(20):
            episode_rewards += agent.play_episode()
        episode_rewards /= 20
        print(f'* Episode{episode},\t Accuracy: {episode_rewards:.2f}')
        if episode_rewards >0.8:
            print(f'Solved in {time.process_time() - start_time}')
            break
#!/usr/bin/env python
# coding: utf-8

import gym
import ptan
import argparse
import time
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F


GAMMA = 0.99
LR = 0.01
WIN_REWARDS = 195
EPISODES_TO_TRAIN = 4

class REL(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, n_actions))

    def forward(self, x):
        return self.layer(x)


def calc_qval(rewards):
    sum_r = 0.0
    res = []
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


class RewardWrapper(ptan.common.wrappers.ClippedRewardsWrapper):
    def reward(self, reward):
        if self.state[0] >= 0.5:
            reward += 101
        elif self.state[0] >= 0:
            reward += self.state[0] + 1
        elif -0.9 >= self.state[0] >= -1.1:
            reward += abs(self.state[0])
        elif self.state[0] < -1.1:
            reward -= 10
        return reward


@torch.no_grad()
def play(env):
    state= env.reset()
    rewards = 0
    while True:
        env.render()
        time.sleep(0.02)
        state_v = torch.FloatTensor([state])
        action = net(state_v).argmax(dim=-1).item()
        state,r,done,_=env.step(action)
        rewards+=r
        if done:
            print(rewards)
            break
    env.close()


env = gym.make('MountainCar-v0')
env = RewardWrapper(env)

net = REL(env.observation_space.shape[0], env.action_space.n)

agent = ptan.agent.PolicyAgent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA)

optimizer = torch.optim.Adam(net.parameters(), lr = LR)

total_rewards = []
cur_reward = 0
episodes = 0
batch_episodes = 0

batch_states, batch_actions, batch_rewards, batch_qval = [],[],[],[]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action="store_true", help='Play and episode when training is complete')
    args = parser.parse_args()
    start = datetime.now()

    for frame_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(exp.action)
        batch_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_episodes += 1
            batch_qval.extend(calc_qval(batch_rewards))
            batch_rewards.clear()
            
        
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = np.mean(total_rewards[-100:])
            print('Step:{}, Reward:{:.0f}, mean_reward:{:.2f}, Episode:{}'.format(frame_idx,reward, mean_rewards,episodes))
            if mean_rewards > WIN_REWARDS:
                print('Solved in {} episodes within {}'.format(episodes, timedelta(seconds=(datetime.now()-start).seconds)))
                break

        
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        actions_v = torch.LongTensor(batch_actions)
        batch_qval_v = torch.FloatTensor(batch_qval)

        logits_v = net(states_v)
        prob_log_v = F.log_softmax(logits_v, dim=1)
        prob_log_actions_v = batch_qval_v * prob_log_v[range(len(batch_states)),actions_v]
        loss_v = -prob_log_actions_v.mean()
        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qval.clear()
        batch_rewards.clear()



    if args.play: play(env)




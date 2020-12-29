# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
import torch
import time
import numpy as np
import torch.nn as nn
from datetime import datetime, timedelta
from torch.nn import functional as F
from collections import namedtuple


hidden_size =128
batch_size = 100
percentile = 70
lr = 0.01


class Net(nn.Module):
    def __init__(self, obs_size, hidden, num_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, num_actions)
                                   )

    def forward(self,x):
        return self.net(x)


Episode = namedtuple('Episode',('reward','steps'))
Steps = namedtuple('Steps',('observation','action'))


@torch.no_grad()
def play(env):
    state = env.reset()
    r = 0
    while True:
        env.render()
        time.sleep(0.01)
        action = net(torch.FloatTensor(state)).argmax(dim=-1).item()
        last_state, reward, done, _ = env.step(action)
        r += reward
        if done:
            print(r)
            break
        state = last_state
    env.close()


def iter_batch(env, net, batch_size):
    batch = []
    obs_action = []
    rewards = 0
    obs = env.reset()
    while True:
        obs_t = torch.FloatTensor([obs])
        action_p_t = F.softmax(net(obs_t),dim=-1)
        action_p = action_p_t.detach().numpy()[0]
        action = np.random.choice(len(action_p),p=action_p)
        step = Steps(obs, action)
        obs_action.append(step)
        next_obs,r,done,_= env.step(action)
        rewards += r
        if done:
            e = Episode(rewards,obs_action)
            batch.append(e)
            obs_action = []
            rewards = 0
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s:s.reward, batch))
    reward_boundry = np.percentile(rewards, percentile)
    rewards_mean = float(np.mean(rewards))
    obs_v = []
    act_v = []
    for reward, step in batch:
        if reward < reward_boundry:
            continue
        obs_v.extend(list(map(lambda s:s.observation, step)))
        act_v.extend(list(map(lambda s:s.action, step)))

    obs_v = torch.FloatTensor(obs_v)
    act_v = torch.LongTensor(act_v)
    return (obs_v, act_v, reward_boundry, rewards_mean)


if __name__=="__main__":
    start_time = datetime.now()
    env = gym.make('CartPole-v1')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    net = Net(obs_size, hidden_size, num_actions)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr= lr)

    for i, batch in enumerate(iter_batch(env, net, batch_size)):
        obs_v, act_v, reward_boundry, rewards_mean = \
            filter_batch(batch, percentile)

        optimizer.zero_grad()
        output = net(obs_v)
        loss = loss_fun(output, act_v)
        loss.backward()
        optimizer.step()

        print(f'epoch:{i} loss:{loss.item():.3f} mean:{rewards_mean:.0f}')

        if rewards_mean > 475:
            duration = timedelta(seconds = (datetime.now()-start_time).seconds)
            print(f'Solved! in {duration}')
            break

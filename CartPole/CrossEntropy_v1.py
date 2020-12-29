# -*- coding: utf-8 -*-
"""
Solve CartPole v1 which rewards is up to 500
Policy approach using CrossEntropy loss trained on
the top 70% episodes

Usually solved in 1:40 sec after playing 31 epochs on Macbook Pro
Solved! in 0:01:41
"""

import gym
import torch
import argparse
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



def iter_batch(env, net, batch_size):
    '''
    plays episodes and record observation/action pairs. Combine them with
    rewards using the defined namedTuples and return the batch when it matchs
    required batch_size
    '''
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
    '''
    Filter top performing episodes using pre-defined criteria. e.g. top 70%
    It works by extracting episodes from the batch, calculate their mean rewards
    select episodes with means in the top percentile, convert observations/actions to 
    tensors and return them along with mean_rewards and boundry
    '''
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='Play an episode after training is complete')
    parser.add_argument('--save', action='store_true', help='Save the trained CNN as crossEntropyV1.dat')
    args = parser.parse_args()

    start_time = datetime.now()
    env = gym.make('CartPole-v1')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    net = Net(obs_size, hidden_size, num_actions)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr= lr)
    '''
    loop over the iter_batch function, which keeps returning batchs of episodes
    filter them to match the percentile, then use that to train the network
    Loss function is the built in CrossEntropy which works very nicely in this case
    '''
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
    
    if args.save: torch.save(net.state_dict(), 'trainedModels/crossEntropyV1.dat')
    if args.play: 
        from lib.common import play_episode
        play_episode(env,net)

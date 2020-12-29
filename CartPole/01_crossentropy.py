# -*- coding: utf-8 -*-
"""
Solve CartPole v0 - top rewards 200
Policy approach using CrossEntropy loss trained on
the top 70% episodes.
steps:
1- write a function to play and return a batch of episodes with (Rewards(observation, action))
2- Filter the batch for the top percentile episodes
3- Train the network using the elite episodes
4- Calculate loss using CrossEntropyLoss function between network output and actions from prime episodes

Note: CrossEntropy method is considered the starting point of policy_gradient methods, where Q(s,a) is 1 for 
the top 70% episodes and 0 for all else.
* CrossEntropy error uses F.log_softmax and NLL

This is a policy method with that generally follows the following steps:
1. Play N number of episodes using our current model and environment.
2. Calculate the total reward for every episode and decide on a reward
boundary. Usually, we use some percentile of all rewards, such as 50th
or 70th.
3. Throw away all episodes with a reward below the boundary.
Chapter 4
[ 87 ]
4. Train on the remaining "elite" episodes using observations as the input and
issued actions as the desired output.
5. Repeat from step 1 until we become satisfied with the result.
Maxim Lapan
"""

import gym
import torch
import numpy as np
import torch.nn as nn
import time
import argparse
from torch.nn import functional as F
from collections import namedtuple


hidden_size =128
batch_size = 32
percentile = 60
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
    parser.add_argument('--play', action='store_true', help ='play an episode after training the network')
    parser.add_argument('--save', action='store_true', help ='save trained network')
    args = parser.parse_args()
    start_time = time.process_time()
    env = gym.make('CartPole-v0')
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
        
        if rewards_mean >= 199:
            print(f'Solved! in {time.process_time()-start_time:.1f}')
            break
    
    if args.save: torch.save(net.state_dict(), 'trainedModels/crossEntropy.dat')
    if args.play: 
        from lib.common import play_episode
        play_episode(env,net)

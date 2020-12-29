#!/usr/bin/env python
# coding: utf-8
'''
Author Ayman Al jabri
Reinforce Method:
Is one of the simplist Policy_gradient methods. It uses the same formula loss= - sum(Q(s,a) log(pi(s,a)))
Where Q(s,a): is the gradient scale. Q(s,a) = discounted rewards or sum(gamm**i * ri)
steps:
    1.Initialize the network with random weights
    2. Play N full episodes, saving their (𝑠,𝑎,𝑟,𝑠′) transitions
    3. For every step, t, of every episode, k, calculate the discounted total reward for
        subsequent steps: 𝑄(𝑘,𝑡) = Σ 𝛾^𝑖 * 𝑟_𝑖
    4. Calculate the loss function for all transitions: ℒ = −Σ𝑄(𝑘,𝑡) log 𝜋(𝑠,𝑎)
    5. Perform an SGD update of weights, minimizing the loss (Use Adam instead - much faster)
    6. Repeat from step 2 until converged

Usually solve in 440 episodes within 0:00:09
'''

import gym
import ptan
import numpy as np
import argparse
from datetime import datetime, timedelta
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class REL(nn.Module):
    '''
    Simple neural network with two linear layers with one ReLU activation. Nothing fancy!
    '''
    def __init__(self, in_features, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions))

    def forward(self, x):
        return self.layer(x)


def calc_qval(rewards):
    '''
    Function to calculate the discounted future rewards
    Takes in list of rewards and discount rate
    Returns the accumlated future values of these rewards
    Example:
    r = [1,1,1,1,1,1]
    gamma = 0.9
    >>> [4.68559, 4.0951, 3.439, 2.71, 1.9, 1.0]
    '''
    sum_r = 0.0
    res = []
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


@torch.no_grad()
def play(env, agent):
    state= env.reset()
    rewards = 0
    while True:
        env.render()
        state_v = torch.FloatTensor([state])
        action = agent(state_v).item()
        state,r,done,_=env.step(action)
        rewards+=r
        if done:
            print(rewards)
            break
    env.close()



if __name__=='__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--play', action='store_true', help='Play an episode after training is complete')
    parser.add_argument('--save',action='store_true', default=False, help='Store a copy of the network')
    args = parser.parse_args()

    GAMMA = 0.99
    LR = 1e-2
    WIN_REWARDS = 195
    EPISODES_TO_TRAIN = 4
    
    env = gym.make('CartPole-v0')
    net = REL(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA)

    'MAKE SURE YOU USE ADAM OPTIMIZER AS IT IS THE MOST STABLE FOR THIS LEARNING ALGORITHM'
    'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    optimizer = torch.optim.Adam(net.parameters(), lr = LR)

    total_rewards = []
    cur_reward = 0
    episodes = 0
    batch_episodes = 0
    loss_v =torch.tensor([0.0])
    batch_states, batch_actions, batch_rewards, batch_qval = [],[],[],[]
    start_training = datetime.now()
    start_print = time()
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
            if time() - start_print > 1:
                print(f'Step:{frame_idx:12}, Loss:{loss_v.item():6.2f}, Reward:{reward:6}, Mean Rewards:{mean_rewards:6.2f}, Episode:{episodes:3}')
                start_print = time()
            if mean_rewards > WIN_REWARDS:
                print('Solved in {} episodes within {}'.format(episodes, timedelta(seconds=(datetime.now()-start_training).seconds)))
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
    
    if args.save: torch.save(net.state_dict(), 'trainedModels/reinforce.dat')
    if args.play: play(env)
    

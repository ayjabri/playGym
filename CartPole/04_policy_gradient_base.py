# -*- coding: utf-8 -*-
'''
Now adays nobody uses plain_vanilla policy gradient but this is for illustration purposes.
Policy gradient: is an improvment of 'Reinforce' method. It addresses two important problems:
a. high gradient variance: it normalizes Q(s,a) value by substracting a baseline from it (the baseline itself
    can be calculated using different methods; average_rewards, moving average or state value. Here we are using average rewards
b. exploration: it improves exploration by punishing the agent when it is too certain of the action. it does so
    by simply substracting entropy value from policy loss

Steps:
1- Initialize the network with random weights.
2- Play N full episodes, saving their (s,a,r,s') transitions
3- For every step t of every episode, calculate the total discounted 
    rewards of subsequent steps 𝑄(𝑘,𝑡) = Σ 𝛾^𝑖 * 𝑟_𝑖 - baseline
4- Calculate policy loss = ℒ = −Σ 𝑄(𝑘,𝑡) log 𝜋(𝑠,𝑎)
5- Perform SGD update of weights
6- Repeat from step 2

Usually solves in 0:00:10 after playing 67 epochs when using the same paramteres 
in this file it should solve in about 8 seconds. just make sure to use ADAM optimizer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import gym
import ptan
from datetime import datetime, timedelta
from tensorboardX import SummaryWriter
from time import time



class Net(nn.Module):
    '''
    Simple neural network with two linear layers with one ReLU activation. Nothing fancy!
    '''
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(obs_shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, n_actions))
    
    def forward(self, x):
        return self.layer(x)


def calc_q_values(rewards, baseline=True):
    if not baseline: return rewards
    baseline = np.cumsum(rewards)/np.arange(1, len(rewards)+1)
    return rewards - baseline


def batch_generator(exp_source, gamma, play_episodes=4, baseline=True):
    '''
    Uses ptan's experience_source object to extract (state, action, reward, last_state) for each episode
    compile them in tensors, calculate discounted rewards in a separate tensor and return with others.
    Example output of an experience step:
    ExperienceFirstLast(state=array([ 0.0159272 ,  0.04763732, -0.03788334, -0.02975388]),
                        action=1,
                        reward=1.0,
                        last_state=array([ 0.01687995,  0.24328149, -0.03847841, -0.33414461]))
    
    '''
    batch_s, batch_a, batch_r = [],[],[]
    t_rewards = []
    episodes = 0
    for exp in exp_source:
        batch_s.append(exp.state)
        batch_a.append(exp.action)
        batch_r.append(exp.reward)

        reward = exp_source.pop_total_rewards()
        if reward:
            t_rewards.append(reward)
            episodes += 1
            if episodes == play_episodes:
                q_values_v = calc_q_values(batch_r, baseline)
                yield (torch.FloatTensor(batch_s),
                    torch.LongTensor(batch_a),
                    torch.FloatTensor(q_values_v),
                    t_rewards)
                batch_s.clear()
                batch_a.clear()
                batch_r.clear()
                t_rewards.clear()
                episodes = 0



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help ='play an episode after training the network')
    parser.add_argument('--save', action='store_true', help ='save trained network')
    parser.add_argument('--baseline', action='store_true', default = False, help ='save trained network')
    args = parser.parse_args()

    GAMMA = 0.98
    LR = 1e-2
    ENTROPY_BETA = 0.01
    BATCH_SIZE = 128
    STEPS = 10 # ptan experiecne calculates the discounted rewards for me. All I need is to calculate the batch baseline 
    PLAY_EPISODES = 10 

    env = gym.make('CartPole-v0')
    net = Net(env.observation_space.shape, env.action_space.n)
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(net, selector, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=STEPS)

    'MAKE SURE YOU USE ADAM OPTIMIZER AS IT IS THE MOST STABLE FOR THIS LEARNING ALGORITHM'
    'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    optimizer = torch.optim.Adam(net.parameters(), lr= LR)

    total_rewards = []
    epoch = 0
    loss = 0.0
    start_time = datetime.now()
    start_print = time()
    print(net)
    for s_v, a_v, disc_r_v, rewards in batch_generator(exp_source, GAMMA, play_episodes= PLAY_EPISODES, baseline=args.baseline):
        epoch += 1
        total_rewards.extend(rewards)
        mean = round(np.mean(total_rewards[-100:]),2)
        if time() - start_print > 1:
            print(f'Epoch:{epoch:3}, Loss: {loss:3.2f}, Reward:{np.mean(rewards):6.2f}, Mean Rewards:{mean:6.2f}')
            start_print = time()
        if mean > 195:
            duration = timedelta(seconds=(datetime.now()-start_time).seconds)
            print(f'Solved in {duration}')
            break
        
        optimizer.zero_grad()
        logit_v = net(s_v)
        log_prob_v = F.log_softmax(logit_v, dim=1)
        prob_v_a = log_prob_v[range(len(a_v)), a_v]
        policy_loss = - (disc_r_v * prob_v_a).mean(dim=-1)

        #entropy loss
        prob_v = F.softmax(logit_v, dim=1)
        ent = - (prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss = ENTROPY_BETA * ent

        loss = policy_loss + entropy_loss
        loss.backward()
        optimizer.step()

    if args.save: torch.save(net.state_dict(), 'trainedModels/policyRL.dat')
    if args.play: 
        from lib.common import play_episode
        play_episode(env,net)

"""
Solving Lunar lander using Plain Vanilla reinforcement method:
    Network: Simple 2 layers with one output returning mu
    Observations: fresh episodes played from experience-source (must complete full episode to start training)
    Rewards: discounted using gamma 
    Loss: is the negative mean log of probability (log_soft), multiplied by discounted rewards

--- training took 6 hours to solve ---
epoch:  2137 mean: 149.81, loss:   8.78, reward 157.65
epoch:  2137 mean: 150.17, loss:   8.78, reward 184.60
Solved in 6:17:35

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import ptan
import argparse
from time import time
from datetime import datetime, timedelta


# =============================================================================
# Simple NN with two layers
# =============================================================================
class Net(nn.Module):
    def __init__(self, shape, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(shape, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, n_actions)
                                   )
        
    def forward(self, x):
        return self.layer(x)

# =============================================================================
# Scale the network using discounted rewards
# =============================================================================
def disc_rewards(rewards, gamma):
    res = 0.0
    l = []
    for e in reversed(rewards):
        res *= gamma
        res += e
        l.append(res)
    return list(reversed(l))

# =============================================================================
# Play function to execute when training is complete
# =============================================================================
@torch.no_grad()
def play(env,agent):
    state = env.reset()
    rewards= 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()  
   
# =============================================================================
# Hyperparameters
# =============================================================================
GAMMA = 0.99
LR = 1e-3
SOLVE = 150


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_dqn.dat"')
    parser.add_argument('--episodes', '-e', default=4, type=int, help='Episodes to put in each training batch')
    args = parser.parse_args()
    env = gym.make('LunarLander-v2')
    net = Net(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=1)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    batch_s, batch_a, batch_r, batch_qval = [],[],[],[]
    total_rewards =[]
    episode = 0
    epoch = 0
    loss = 0.0
    start = time()
    print(net)
    start_time = datetime.now()
    for idx,exp in enumerate(exp_source):
        batch_s.append(exp.state)
        batch_a.append(exp.action)
        batch_r.append(exp.reward)

        reward = exp_source.pop_total_rewards()
        if reward:
            batch_qval.extend(disc_rewards(batch_r, GAMMA))
            batch_r.clear()
            episode += 1
            total_rewards.append(reward[0])
            mean = np.mean(total_rewards[-100:])
            if time()-start >1:
                print(f'epoch:{epoch:6} mean:{mean:7.2f}, loss:{loss:7.2f}, reward{reward[0]:7.2f}')
                start = time()
            if mean > SOLVE:
                duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                if args.save: torch.save(net.state_dict(),'lunar_rl.dat')
                if args.play: play(env,agent)
                break
        
        if episode < args.episodes:
            continue
        epoch += 1
        
        state_v = torch.FloatTensor(batch_s)
        act_v = torch.LongTensor(batch_a)
        qval_v = torch.FloatTensor(batch_qval)

        optimizer.zero_grad()
        logit_v = net(state_v)
        log_prob_v = F.log_softmax(logit_v, dim=1)
        log_prob_a_v = log_prob_v[range(len(act_v)),act_v]
        loss = - (log_prob_a_v * qval_v).mean()
        loss.backward()
        optimizer.step()
        
        batch_s.clear()
        batch_a.clear()
        batch_r.clear()
        batch_qval.clear()
        episode = 0



# -*- coding: utf-8 -*-
"""
Deep Q Learning approach
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import argparse
import ptan
import numpy as np
from datetime import datetime, timedelta


class Net(nn.Module):
    def __init__(self, obs_shape, hidden_size, n_actions):
        super(Net, self).__init__()
        self.obs_shape = obs_shape
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.layer = nn.Sequential(nn.Linear(obs_shape[0], hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_actions))
        
        self.move()
        
    def forward(self, x):
        return self.layer(x.float())
    
    def move(self):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    


def unpack_batch(batch):
    s,a,r,d,l = [],[],[],[],[]
    for exp in batch:
        s.append(exp.state)
        a.append(exp.action)
        r.append(exp.reward)
        d.append(exp.last_state is None)
        if exp.last_state is not None:
            l.append(exp.last_state)
        else:
            l.append(exp.state)
    return (np.array(s, dtype=np.float32),
            np.array(a, dtype=np.long),
            np.array(r),
            np.array(d),
            np.array(l, dtype=np.float32))


def calc_loss(batch, net, tgt_net, gamma):
    s,a,r,d,l = unpack_batch(batch)
    states = torch.FloatTensor(s)
    actions = torch.LongTensor(a)
    rewards = torch.FloatTensor(r)
    dones = torch.BoolTensor(d)
    last_states = torch.FloatTensor(l)
    
    # q_sa = net(states)[range(len(a)), actions]
    q_sa = net(states).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        '''
        Y = r + Gamma * max (Q(s',a'))
        '''
        q_sa_prime = tgt_net.target_model(last_states).max(dim=1)[0]
        q_sa_prime[dones] = 0
        y = rewards + gamma * q_sa_prime
    return F.mse_loss(q_sa, y)


if __name__=='__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--play', action='store_true', help='Play an episode after training is complete')
    parser.add_argument('--save',action='store_true', default=False, help='Store a copy of the network')
    args = parser.parse_args()
    env = gym.make('CartPole-v0')
    net = Net(env.observation_space.shape, 128, env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)
    GAMMA = 0.96

    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, eps_start=1.0, eps_final = 0.02, eps_frames= 10_000)
    agent = ptan.agent.DQNAgent(net, selector, device= net.device, preprocessor= ptan.agent.float32_preprocessor)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, 20_000)

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
    
    total_rewards = []
    frame_idx = 0
    episode = 0
    start_time = datetime.now()
    while True:
        frame_idx += 1
        eps_tracker.frame(frame_idx)
        
        buffer.populate(1)
        
        reward = exp_source.pop_total_rewards()
        
        if reward:
            episode += 1
            total_rewards.append(reward)
            mean = np.mean(total_rewards[-100:])
            print(f'Mean:{mean:.2f}, epsilone:{selector.epsilon:.2f}')
            if mean > 195:
                duration = timedelta(seconds= (datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                break
        
        if len(buffer) < 2000:
            continue
        
        optimizer.zero_grad()
        batch = buffer.sample(500)
        loss = calc_loss(batch, net, tgt_net, gamma=0.98)
        loss.backward()
        optimizer.step()
        
        if frame_idx %1000 == 0:
            tgt_net.sync()
    
    if args.save: torch.save(net.state_dict(), 'trainedModels/TR.dat')
    if args.play: 
        from lib.common import play_episode
        play_episode(env,net)

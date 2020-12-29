# -*- coding: utf-8 -*-
"""
Spyder Editor

Author@ Ayman Al Jabri
Solve BipedalWalker using Deep Deterministic Policy Gradient (DDPG) method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import ptan
import numpy as np
from time import time
from datetime import datetime
from collections import deque
from math import inf


class DDPGActorNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, 400),
                                  nn.ReLU(),
                                  nn.Linear(400, 300),
                                  nn.ReLU(),
                                  nn.Linear(300, act_size),
                                  nn.Tanh(),
                                  )
    def forward(self, x):
        return self.base(x)
    
    
class DDPGCriticNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.net_in = nn.Sequential(nn.Linear(obs_size, 400),
                                    nn.ReLU(),
                                    )
        self.q_val = nn.Sequential(nn.Linear(400 + act_size, 300),
                                   nn.ReLU(),
                                   nn.Linear(300, 1),
                                   )
    def forward(self, s, a):
        base = self.net_in(s)
        return self.q_val(torch.cat([base, a], dim=1))
    
    
class AgentDDPG(ptan.agent.BaseAgent):
    def __init__(self, model, device='cpu', epsilon=0.3):
        self.model = model
        self.device = device
        self.epsilon = epsilon
    
    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_np = self.model(states_v).data.cpu().numpy()
        mu_np += np.random.normal(size=mu_np.size) * self.epsilon
        return np.clip(mu_np, -1, 1), agent_states


def unpack_dqn(batch, device='cpu'):
    states, actions, rewards, dones, last_states =\
        [],[],[],[],[]
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_v = torch.FloatTensor(np.array(actions, copy=False)).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.BoolTensor(dones).to(device)
    last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
    return (states_v, actions_v, rewards_v, dones, last_states_v)


@torch.no_grad()
def play(env,act_net):
    obs = env.reset()
    rewards = 0.0
    while True:
        env.render()
        obs_v = ptan.agent.float32_preprocessor(obs)
        action = act_net(obs_v).data.numpy()
        obs,r,done,_=env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()
    
        
if __name__=='__main__':
    ENV_ID = 'BipedalWalker-v3'
    ACT_LR = 5e-5
    CRT_LR = 4e-4
    GAMMA = 0.99
    STEPS = 5
    BUFFER_SIZE = 100_000
    BOUND = 250
    INIT_REPLAY = 20_000
    BATCH_SIZE = 128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(ENV_ID)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    
    actor_net = DDPGActorNet(obs_size, act_size).to(device)
    tgt_actor_net = ptan.agent.TargetNet(actor_net)
    critic_net = DDPGCriticNet(obs_size, act_size).to(device)
    tgt_critic_net = ptan.agent.TargetNet(critic_net)
    print(actor_net, critic_net)
    
    act_optim = torch.optim.Adam(actor_net.parameters(),ACT_LR)
    crt_optim = torch.optim.Adam(critic_net.parameters(),CRT_LR)
    
    agent = AgentDDPG(actor_net)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent,\
                                               GAMMA, steps_count=STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, BUFFER_SIZE)
    
    frame_idx = 0
    last_frame = 0
    episode = 0
    actor_loss = 0.0
    total_rewards = deque(maxlen=100)
    pt = time()
    st = datetime.now()
    top_reward = -inf
    
    while True:
        frame_idx += 1
        buffer.populate(1)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            episode += 1
            total_rewards.append(new_reward)
            mean=np.mean(total_rewards)
            if mean>BOUND:
                print('Solved!')
                break
            if time()-pt > 2:
                fps = (frame_idx - last_frame) / (time()-pt)
                print(f'{frame_idx:7,}: episode:{episode:6}, mean:{mean:7.2f}, actor loss:{actor_loss:7.2f}, speed: {fps:7.2f} fps')
                pt = time()
                last_frame = frame_idx
            if mean > top_reward:
                print('*** New Top Reward ***')
                fname = ENV_ID + '_new_top.dat'
                torch.save(actor_net.state_dict(),fname)
                top_reward = mean
        if len(buffer)<INIT_REPLAY:
            continue
        if len(buffer)==INIT_REPLAY:print(15*'*','Training Started',15*'*')
        
        batch = buffer.sample(BATCH_SIZE)
        states, actions, rewards, dones, last_states = unpack_dqn(batch)
        
        # Train Critic
        crt_optim.zero_grad()
        q_sa = critic_net(states, actions)
        a_last = tgt_actor_net.target_model(last_states)
        q_sa_last = tgt_critic_net.target_model(last_states, a_last)
        q_sa_last[dones] = 0.0
        q_ref_val = rewards.unsqueeze(-1) + q_sa_last * GAMMA
        critic_loss = F.mse_loss(q_sa, q_ref_val.detach())
        critic_loss.backward()
        crt_optim.step()
        
        # Train Actor
        act_optim.zero_grad()
        a_curr = actor_net(states)
        actor_loss = (- critic_net(states, a_curr)).mean()
        actor_loss.backward()
        act_optim.step()
        
        tgt_actor_net.alpha_sync(alpha=1-1e-3)
        tgt_critic_net.alpha_sync(alpha=1-1e-3)
        
        

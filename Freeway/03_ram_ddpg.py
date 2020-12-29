#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:06:46 2020

@author: ayman
"""

import gym
import ptan
import torch
import torch.nn.functional as F
import numpy as np
from lib import model, utils
from tensorboardX import SummaryWriter


if __name__ =='__main__':
    params = model.HYPERPARAMS['cartpole']
    N_ENVS = 1
    envs = []
    for _ in range(N_ENVS):
        env = gym.make(params.ENV_ID)
        env.seed(params.SEED)
        envs.append(env)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    
    act_net = model.DDPGActor(obs_size, act_size)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    crt_net = model.DDPGCritic(obs_size, act_size)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    
    act_optim = torch.optim.Adam(act_net.parameters(), lr=1e-3)
    crt_optim = torch.optim.Adam(crt_net.parameters(), lr=1e-2)
    
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = model.DDPGAgent(act_net, selector)
    eps_tracker = model.EpsilonTracker(selector,params.EPS_START,params.EPS_END,
                                       params.EPS_START_FRAME, params.EPS_END_FRAME)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.GAMMA,
                                                           steps_count=params.STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.BUFFER_SIZE)
    comment = (f'_ddpg_{params.STEPS}_{len(envs)}')
    writer = SummaryWriter(comment=comment)
    frame = 0.0
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += N_ENVS
            eps_tracker.frame(frame)
            buffer.populate(N_ENVS)
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(new_reward, frame, epsilon=selector.epsilon)
                if mean:
                    if mean > params.SOLVE:
                        print('Solved')
                        break
            if len(buffer) < params.INIT_REPLAY:
                continue
            
            batch = buffer.sample(params.BATCH_SIZE * N_ENVS)
            states, actions, rewards, dones, last_states = utils.unpack_batch(batch)
            
            states_v = torch.FloatTensor(states)
            actions_v = torch.FloatTensor(actions).unsqueeze(-1)
            last_states_v = torch.FloatTensor(last_states)
            rewards_v = torch.tensor(rewards)
            
            # train critic
            crt_optim.zero_grad()
            qval_s_a = crt_net(states_v, actions_v)
            tgt_act_output = tgt_act_net.target_model(last_states_v)
            next_actions = tgt_act_net.target_model.apply_softmax(tgt_act_output).argmax(dim=1).unsqueeze(-1)
            
            next_qval = tgt_crt_net.target_model(last_states_v, next_actions)
            next_qval[dones] = 0.0
            q_ref_v = rewards_v.unsqueeze(-1) + next_qval * params.GAMMA
            crt_loss = F.mse_loss(qval_s_a, q_ref_v.detach())
            crt_loss.backward()
            crt_optim.step()
            
            # train actor
            act_optim.zero_grad()
            curr_act = act_net.apply_softmax(act_net(states_v)).argmax(dim=1).unsqueeze(-1)
            act_loss = - (crt_net(states_v, curr_act)).mean()
            act_loss.backward()
            act_optim.step()
            
            tgt_act_net.alpha_sync(alpha=1-1e-3)
            tgt_crt_net.alpha_sync(alpha=1-1e-3)
            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 08:43:29 2020

@author: Ayman Al Jabri
"""
import torch
import gym
import ptan
from tensorboardX import SummaryWriter
from lib import model, utils
import argparse
from collections import deque



params = model.HYPERPARAMS['freeway']


@torch.no_grad()
def play_dqn(env, net):
    state = env.reset()
    rewards = 0.0
    while True:
        env.render()
        state_v = ptan.agent.float32_preprocessor([state])
        action = net(state_v).argmax(dim=-1)
        state,reward,done,_=env.step(action)
        rewards+=reward
        if done:
            print(rewards)
            break
    env.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', default=params.STEPS, type=int, help='Number of steps to skip when extracting\
                        sample transitions')
    parser.add_argument('--envs', default=4,type=int, help='Number of environments to \
                        extract observations from at the same time' )
    args = parser.parse_args()
    
    
    envs = []
    for _ in range(args.envs):
        env = gym.make(params.ENV_ID)
        env.seed(params.SEED)
        envs.append(env)
    
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    
    net = model.DQNNet(obs_size, act_size)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = model.EpsilonTracker(selector,eps_start=params.EPS_START,
            eps_final=params.EPS_END, eps_start_frame=params.EPS_START_FRAME,
            eps_end_frame= params.EPS_END_FRAME)
    agent = ptan.agent.DQNAgent(net, selector,preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.GAMMA, steps_count= args.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.BUFFER_SIZE)
    writer = SummaryWriter(comment=f'_{args.steps}_{args.envs}')
    optimizer = torch.optim.Adam(net.parameters(), lr=params.LR)
    
    frame = 0
    total_rewards = deque(maxlen=100)
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += 1
            eps_tracker.frame(frame)
            buffer.populate(1)
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(new_reward[0], frame, selector.epsilon)
                if mean:
                    if mean > params.SOLVE:
                        print('Solved')
                        break
            if len(buffer) < params.INIT_REPLAY:
                continue

            batch = buffer.sample(params.BATCH_SIZE * args.envs)
            optimizer.zero_grad()
            loss = utils.calc_dqn_loss(batch, net, tgt_net, params.GAMMA)
            loss.backward()
            optimizer.step()
            
            if frame % 1000==0:
                tgt_net.sync()
            

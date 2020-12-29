#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:03:29 2020
@author: Ayman Al Jabri

Steps:
1- Class network with two heads (val: 1 output that estimates V(s)) and (policy: n_actions output that esitmates the policy pi(a,s))
2- Initialize network parameteres with random values
3- Play N steps in the environment using the current policy and saving state, action, reward
4- for i = t-1...t (steps are in reversed order):
    a. calculate discounted rewards in reversed order: ri+gamma*R -> R (ptan library calculates this for us for N steps)
    b. Represent total rewards as Q(s,a) = V(s) + Adv(s,a) -> Adv(s,a) = Q(s,a) - V(s)
        First: calc Q(s,a) = Sum_{0 to N-1} GAMMA^i * r_i + GAMMA^N * V(s_N) ---- V(s_N) is the value head of our network when fed last_state
        second: calc Adv(s,a) = Q(s,a) - V(s) -> use to scale policy loss 
    c. accumulate policy gradients: - Σ Adv(s,a) * log(pi(a,s))* (R - V(si)) -> policy_grad
    d. accumulate value gradients: value_grad + MeanSquareError(R, V(si))
5- update the network parameters using the accumulated gradients, moving in the direction of policy gradient and opposite to value gradient (i.e. subtract policy and add value!)
6- repeat from step 2 until convergence

It should solve in about 29 seconds after playing 184 epochs ... it varies a lot though!
Solved in 0:00:29 seconds!

"""
import gym
import ptan
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import numpy as np
from time import time
from datetime import datetime, timedelta
from ptan.common import wrappers


class A2CNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, 256),
                                  nn.ReLU())
        self.policy = nn.Linear(256, act_size)
        self.val = nn.Linear(256, 1)

    def forward(self, x):
        base_output = self.base(x)
        return self.policy(base_output), self.val(base_output)


def unpack_batch(batch, net):
    states, actions, rewards, last_states = [], [], [], []
    not_done_idx = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            last_states.append(np.array(exp.last_state, copy=False))
            not_done_idx.append(idx)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_t = torch.FloatTensor(np.array(last_states, copy=False))
        last_values_np = net(last_states_t)[1].data.cpu().numpy()
        last_values_np *= GAMMA**STEPS
        rewards_np[not_done_idx] += last_values_np[:, 0]
    ref_val_t = torch.FloatTensor(rewards_np)
    states_t = torch.FloatTensor(np.array(states, copy=False))
    actions_t = torch.tensor(actions)
    return states_t, actions_t, ref_val_t


def calc_losses(batch, net):
    s_t, a_t, ref_v = unpack_batch(batch, net)
    logits_v, val_v = net(s_t)
    # Calculate value loss
    value_loss = F.mse_loss(val_v.squeeze(-1), ref_v)
    # Calculate Policy Loss
    log_prob_v = F.log_softmax(logits_v, dim=1)
    log_prob_a_v = log_prob_v[range(len(a_t)), a_t]
    adv_v = ref_v - val_v.squeeze(-1).detach()
    policy_loss_v = adv_v * log_prob_a_v
    policy_loss = - policy_loss_v.mean()
    # Calc Entropy loss
    prob_v = F.softmax(logits_v, dim=1)
    ent = (prob_v * log_prob_v).sum(dim=1).mean()
    entropy_loss = ENTROPY_BETA * ent

    return value_loss, policy_loss, entropy_loss


@torch.no_grad()
def play(env, agent):
    state = env.reset()
    rewards = 0.0
    while True:
        env.render()
        state_v = ptan.agent.float32_preprocessor([state])
        action, _ = agent(state_v)
        state, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',
                        help='Play an episode after training is complete')
    parser.add_argument('--save', action='store_true',
                        default=False, help='Store a copy of the network')
    args = parser.parse_args()

    GAMMA = 0.99
    STEPS = 1
    LR = 1e-4
    ENVS = 5
    ENV_ID = 'Freeway-ram-v0'
    ENTROPY_BETA = 0.02
    CLIP_GRAD = 0.1
    SOLVE_REWARD = 20
    PLAY_EPISODES = 1

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    float32 = ptan.agent.float32_preprocessor

    envs = []
    for _ in range(ENVS):
        env = gym.make(ENV_ID)
        env = wrappers.NoopResetEnv(env)
        envs.append(env)

    obs_size = envs[0].observation_space.shape[0]
    act_size = envs[0].action_space.n

    net = A2CNet(obs_size, act_size)
    agent = ptan.agent.ActorCriticAgent(net, device=device, apply_softmax=True,
                                        preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, GAMMA, steps_count=STEPS)
    optimizer = torch.optim.Adam(net.parameters(), LR)

    total_rewards = []
    batch = []
    print_time = time()
    start_training = datetime.now()
    episode = 0
    epoch = 0
    train_episode = 0
    loss_v = torch.tensor([0.0])
    for exp in exp_source:
        batch.append(exp)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            train_episode += 1
            episode += 1
            total_rewards.append(new_reward[0])
            mean = np.mean(total_rewards[-100:])
            delta_time = time() - print_time
            if delta_time > 2:
                print(
                    f'Epoch:{epoch:3}, Loss: {loss_v.item():3.2f}, Reward:{new_reward[0]:6.2f}, Mean Rewards:{mean:6.2f}')
                print_time = time()
            if mean > SOLVE_REWARD:
                done_training = datetime.now()
                training_time = timedelta(
                    seconds=(done_training-start_training).seconds)
                print(
                    f'Epoch:{epoch:3}, Loss: {loss_v:6.2f}, Reward:{new_reward[0]:6.2f}, Mean Rewards:{mean:6.2f}')
                print(
                    f'Solved in {training_time} seconds after seeling {len(total_rewards)} episodes!')
                break

        if train_episode < PLAY_EPISODES:
            continue
        epoch += 1
        optimizer.zero_grad()
        value_loss, policy_loss, entropy_loss = calc_losses(batch, net)
        batch.clear()
        loss_v = policy_loss + value_loss + entropy_loss
        nn_utils.clip_grad_norm_(net.parameters(), max_norm=CLIP_GRAD)
        loss_v.backward()
        optimizer.step()
        train_episode = 0

    if args.save:
        torch.save(net.state_dict(), ENV_ID + '_a2c.dat')
    if args.play:
        play(envs[0], agent)

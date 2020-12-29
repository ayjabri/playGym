# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:36:47 2020

@author: ayjab
"""


import gym
import math
import ptan
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


params = SimpleNamespace(**{'env_name':'MsPacmanNoFrameskip-v4',
                            'replay_size':100_000,
                            'init_size': 1000,
                            'eps_start': 1.0,
                            'eps_final': 0.02,
                            'eps_frames': 100_000,
                            'batch_size': 32,
                            'gamma': 0.9,
                            'lr': 1e-3,
                            'sync_freq':1000,
                            'solve_reward':500
                            })


class DuelDQN(nn.Module):
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]
        self.fc_adv = nn.Sequential(nn.Linear(outshape, 256),
                                nn.ReLU(),
                                nn.Linear(256, n_actions))
        self.fc_val = nn.Sequential(nn.Linear(outshape, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))

    def forward(self, x):
        x = self.conv(x.float())
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return (val + (adv - adv.mean(dim=1, keepdim=True)))


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            b = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(b)
            self.register_buffer('epsilon_bias', torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.data.normal_()
        w = self.weight + self.sigma_weight * self.epsilon_weight.data
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, w, bias)


class DuelNoisyDQN(nn.Module):
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]

        self.fc_adv = nn.Sequential(NoisyLinear(outshape, 256),
                                nn.ReLU(),
                                NoisyLinear(256, n_actions))
        self.fc_val = nn.Sequential(NoisyLinear(outshape, 256),
                                nn.ReLU(),
                                NoisyLinear(256, 1))

    def forward(self, x):
        x = self.conv(x.float())
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return (val + (adv - adv.mean(dim=1, keepdim=True)))


def unpack_batch(batch):
    states, actions, rewards, dones, next_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            next_states.append(state)
        else:
            next_state = np.array(exp.last_state)
            next_states.append(next_state)
    return (np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(dones, dtype=np.bool),
            np.array(next_states))


def calc_loss_dqn(batch, net, tgt_net, gamma, device='cpu'):
    states,actions,rewards,dones,next_states = unpack_batch(batch)

    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    dones_v = torch.BoolTensor(dones).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)

    state_action_value = net(states_v).gather(dim=1, index=actions_v.unsqueeze(1)).squeeze(-1)
    with torch.no_grad():
        next_max_q_value = tgt_net(next_states_v).max(dim=1)[0]
        next_max_q_value[dones_v] = 0.0
        expected_q_value = rewards_v + gamma * next_max_q_value

    return F.mse_loss(state_action_value, expected_q_value)




SEED = 120
np.random.seed(SEED)
torch.manual_seed(SEED)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-envs', type=int, default=1, help='Number of environments')
    parser.add_argument('-n', type=int, default=1, help='Number of steps')
    args = parser.parse_args()

    envs =[]
    for _ in range(args.envs):
        env = gym.make(params.env_name)
        env = ptan.common.wrappers.wrap_dqn(env)
        env.seed(SEED)
        envs.append(env)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    f_shape = env.observation_space.shape
    n_actions = env.action_space.n

    net = DuelDQN(f_shape, n_actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(params.eps_start)
    agent = ptan.agent.DQNAgent(net, selector, device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma, steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.replay_size)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start, params.eps_final, params.eps_frames)

    optimizer = torch.optim.SGD(net.parameters(), lr= params.lr)


    frame_idx = 0
    total_rewards = []
    episode = 0
    done_reward = None

    while True:
        frame_idx += 1
        buffer.populate(1)
        eps_tracker.frame(frame_idx)


        done_reward = exp_source.pop_total_rewards()
        if done_reward:
            episode += 1
            total_rewards.append(done_reward)
            mean = np.mean(total_rewards[-50:])
            print(f'{frame_idx}\t episode:{episode}, reward={done_reward[0]}, mean={mean:.3f}, epsilon={selector.epsilon:.3f}')

            if mean > params.solve_reward:
                print(f'Solved in {episode} episodes')
                tgt_net.sync()
                torch.save(tgt_net.target_model.state_dict(),f'{params.env_name}_{len(envs)}_envs_{args.n}_steps.dat')
                break

        if len(buffer) < params.init_size:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params.batch_size)
        loss = calc_loss_dqn(batch, net, tgt_net.target_model, params.gamma, device)
        loss.backward()
        optimizer.step()

        if frame_idx % params.sync_freq == 0:
            tgt_net.sync()


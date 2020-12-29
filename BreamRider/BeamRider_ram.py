#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:31:41 2020

@author: ayman jabri
"""
#%%
import math, torch, gym, argparse, ptan, time, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from tensorboardX import SummaryWriter


class DQN(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(in_features, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, n_actions))
    def forward(self, x):
        return self.layer(x.float())



class DuelDQN(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()

        self.input = nn.Linear(in_features, 256)
        self.fc_adv = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.input(x.float()))
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return (val + (adv - adv.mean(dim=-1, keepdim=True)))


class NoisyLinear(nn.Linear):
    def __init__(self,in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer('epsilon_weight', z)
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3/in_features)
        self.weight.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        w = self.weight + self.sigma_weight * self.epsilon_weight.data
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = self.bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, w, bias)


class NoisyDQN(nn.Module):
    def __init__(self,in_features, n_actions):
        super().__init__()
        self.layer = nn.Sequential(NoisyLinear(in_features, 128, bias=True),
                                   nn.ReLU(),
                                   NoisyLinear(128, n_actions, bias=True))

    def forward(self,x):
        return self.layer(x.float())


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


def calc_loss(batch, net, tgt_net, gamma, device='cpu'):
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


@torch.no_grad()
def play(env, net):
    state = env.reset()
    rewards = 0.0
    while True:
        env.render()
        time.sleep(0.01)
        stateV = torch.FloatTensor(state)
        action = net(stateV).argmax(dim=-1).item()
        next_state,reward,done,_= env.step(action)
        rewards += reward
        if done:
            print(rewards)
            break
        state = np.array(next_state)
    time.sleep(1)
    env.close()




params = SimpleNamespace(**{'env_name':'BeamRider-ram-v0',
                            'run_name':'BreamRider',
                            'replay_size':10000,
                            'init_size': 64,
                            'eps_start': 1.0,
                            'eps_final': 0.02,
                            'eps_frames': 50000,
                            'batch_size': 32,
                            'gamma': 0.95,
                            'lr': 1e-3,
                            'sync_freq':1000,
                            'solve_reward':5000
                            })



#%%

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', default=False,action='store_true', help='Enable Noisy Network')
    parser.add_argument('-n', default=1, type=int, help='Specify n steps')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(params.env_name)
    in_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net=(NoisyDQN(in_features, n_actions) if args.noisy else DuelDQN(in_features, n_actions))
    net = net.to(device)

    # net=DuelDQN(in_features, n_actions)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(params.eps_start, selector=selector)
    agent = ptan.agent.DQNAgent(net, selector,device=device)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.eps_start,\
                                              params.eps_final, params.eps_frames)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, params.gamma, steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.replay_size)
    writer = SummaryWriter(comment=params.run_name)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    stateV = torch.FloatTensor([env.reset()]).to(device)

    writer.add_graph(net,stateV)

    step=0
    episode = 0
    total_rewards = []
#%%
    while True:
        step += 1
        buffer.populate(1)
        eps_tracker.frame(step)
        done_reward = exp_source.pop_total_rewards()
        if done_reward:
            episode += 1
            total_rewards.append(done_reward)
            mean_rewards = np.mean(total_rewards[-50:])
            writer.add_scalar('Episode Reward', done_reward, episode)
            writer.add_scalar('Mean Rewards', mean_rewards, global_step=episode)
            print(f'{step}:episode={episode}, reward={done_reward[0]:.3f}, mean={mean_rewards:.3f}, epsilon={selector.epsilon:.3f}')
            if mean_rewards > params.solve_reward:
                print(f'Solved in {episode} episodes')
                tgt_net.sync()
                torch.save(tgt_net.target_model.state_dict(),f'MountainCar_{args.noisy}_{args.n}.dat')
                play(env, tgt_net.target_model)
                break

        if len(buffer) < params.init_size:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params.batch_size)
        loss = calc_loss(batch, net, tgt_net.target_model, params.gamma**args.n, device=device)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss', loss.item(),step)

        if step % params.sync_freq ==0:
            tgt_net.sync()

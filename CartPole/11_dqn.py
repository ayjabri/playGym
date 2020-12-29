
''' 
After so many trials. The best performace is when
Network: either noraml or Duel (noisy is just all over the place!)
'cartpole': SimpleNamespace(**{
        'env_name': "CartPole-v0",
        'stop_reward': 180.0,
        'run_name': 'CartPole',
        'replay_size': 1000,
        'replay_initial': 64,
        'target_net_sync': 10,
        'epsilon_frames': 5000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 1e-2,
        'gamma': 0.95,
        'batch_size': 32
    })
step-count = 4
gamma**4
environments = 2
batch size = batch-size * len(env)

Solved in 114 Episodes, 11k frames seen
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gym
import ptan
import math

from time import time
from tensorboardX import SummaryWriter
from lib import common
from datetime import datetime, timedelta
import argparse


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init= 0.017, bias=True):
        super(NoisyLinear,self).__init__(in_features, out_features,bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            
            self.sigma_bais = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)
    def forward(self, input):
        self.epsilon_weight.data.normal_()
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.data.normal_()
            bias = bias + self.sigma_bais * self.epsilon_bias.data
        return F.linear(input, weight, bias)


class NoisyDQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.fc1 = NoisyLinear(n_obs, 256)
        self.fc2 = NoisyLinear(256, n_actions)
    
    def forward(self,x):
        x = F.relu(self.fc1(x.float()))
        return self.fc2(x)


class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        return self.fc2(x)


class DuelDQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc_adv = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return (val + (adv - adv.mean(dim=1, keepdim=True)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help ='play an episode after training the network')
    parser.add_argument('--save', action='store_true', help ='save trained network')
    parser.add_argument('--model', default='duel', help ='type of NN to use. Options: dqn, duel, noisy')
    args = parser.parse_args()

    SEED = 123
    N_ENVS = 4
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    params = common.HYPERPARAMS.cartpole

    env = gym.make(params.env_name)
    env.seed(SEED)

    envs = []
    for _ in range(N_ENVS):
        env = gym.make(params.env_name)
        env.seed(SEED)
        envs.append(env)
        
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    if args.model == "duel":
        net = DuelDQN(n_obs, n_actions)
    elif args.model == "noisy":
        net = NoisyDQN(n_obs, n_actions)
    else:
        net = DQN(n_obs, n_actions)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)
    
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(params.epsilon_start, selector)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.epsilon_start, params.epsilon_final, params.epsilon_frames)
    agent = ptan.agent.DQNAgent(net, selector, device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma, steps_count=4)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.replay_size)
    optimizer = torch.optim.SGD(net.parameters(), lr = params.learning_rate)


    writer = SummaryWriter(comment=f'_256_4steps_{params.batch_size}_{params.gamma}_{len(envs)}_envs_'+ params.env_name)
    stateV = torch.FloatTensor([env.reset()])
    writer.add_graph(net, stateV)

    frame_idx = 0
    episodes = 0
    total_rewards = []
    done_rewards = None


    start_time = datetime.now()
    print_time = time()
    while True:
        frame_idx += 1
        buffer.populate(N_ENVS)
        eps_tracker.frame(frame_idx)
        done_rewards = exp_source.pop_total_rewards()

        if done_rewards:
            episodes += 1
            total_rewards.append(done_rewards)
            mean = np.mean(total_rewards[-50:])
            if time() - print_time > 1:
                print(f'{frame_idx:7}: episode={episodes:6}, reward={done_rewards[0]:6}, mean={mean:7.2f}, epsilon={selector.epsilon:6.3f}')
                print_time = time()
            writer.add_scalar('Reward', done_rewards, global_step=episodes)
            writer.add_scalar('Mean Rewards', mean, global_step=episodes)

            if mean > params.stop_reward:
                duration = timedelta(seconds= (datetime.now()-start_time).seconds)
                print(f'Solved in {episodes} within {duration}')
                tgt_net.sync()
                break
        
        if len(buffer) < params.replay_initial:
            continue

        
        optimizer.zero_grad()
        batch = buffer.sample(params.batch_size * len(envs))
        loss = common.calc_loss_dqn(batch, net, tgt_net.target_model, params.gamma**4, device)
        loss.backward()
        optimizer.step()

        if frame_idx % params.target_net_sync == 0:
            tgt_net.sync()
            for name, parameter in net.named_parameters():
                writer.add_histogram(name, parameter)

    if args.save: torch.save(net.state_dict(), 'trainedModels/DQN.dat')
    if args.play: 
        from lib.common import play_episode
        play_episode(env,net)

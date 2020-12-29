'''
Worst performance ever!
I'm not liking A2C or A3C for that matter
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.utils as nn_utils

import gym
import ptan
import os
import math
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from collections import namedtuple


GAMMA = 0.99
STEPS = 4
SEED = 120
ENV_ID = 'Pendulum-v0'
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1
ENVS = 5
TRAIN_EPISODES = 10
SOLVE_BOUND = 150
LR = 1e-4



class A3CNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super(A3CNet, self).__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, 256),
                                nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(256, act_size),
                                nn.Tanh())
        self.var = nn.Sequential(nn.Linear(256, act_size),
                                nn.Softplus())
        self.val = nn.Linear(256,1)

    def forward(self, x):
        base = self.base(x)
        return self.mu(base),self.var(base),self.val(base)


class A3CAgent(ptan.agent.BaseAgent):
    def __init__(self, model,device='cpu',preprocessor=None):
        self.model = model
        self.device = device
        if preprocessor is None:
            self.preprocessor = ptan.agent.float32_preprocessor
        else:
            self.preprocessor = preprocessor
    def __call__(self, state, agent_status=None):
        state_v = self.preprocessor(state).to(self.device)
        mu,var,_ = self.model(state_v)
        mu_np = mu.data.cpu().numpy()
        std_np = np.sqrt(var.data.cpu().numpy())
        rand_a = np.random.normal(mu_np, std_np)
        actions = np.clip(rand_a,-2,2)
        return list(actions), agent_status


def unpack_batch(batch, net, gamma, steps, device='cpu'):
    s,a,r,ls = [],[],[],[]
    not_done_idx = []
    for idx,exp in enumerate(batch):
        s.append(exp.state)
        a.append(exp.action)
        r.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            ls.append(exp.last_state)
    r_np = np.array(r, copy=False, dtype=np.float32)
    if not_done_idx:
        ls_v = torch.FloatTensor(np.array(ls, copy=False)).to(device)
        q_val_np = net(ls_v)[2].data.cpu().numpy()[:,0]
        q_val_np *= gamma**steps
        r_np[not_done_idx] += q_val_np
    q_ref_val = torch.FloatTensor(r_np)
    s_v = torch.FloatTensor(np.array(s, copy=False))
    a_v = torch.FloatTensor(np.array(a, copy=False))
    return s_v, a_v, q_ref_val



def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v



def calc_log_prob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2

@torch.no_grad()
def test_net(net, env, count=5, device="cpu"):
    test_rewards = []
    test_steps = []
    for _ in range(count):
        rewards = 0.0
        steps = 0
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -2, 2)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                test_rewards.append(rewards)
                test_steps.append(steps)
                break
    mean_rewards = np.mean(test_rewards)
    mean_steps = np.mean(test_steps)
    test_rewards.clear()
    test_steps.clear()
    return mean_rewards, mean_steps


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='Play and episode when training is complete')
    parser.add_argument('--save', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_a3c.dat"')
    parser.add_argument('--steps','-s', default=10, type=int, help='Steps used to discount reward') # This is new in Policy Gradient Method
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    envs = []
    for _ in range(ENVS):
        env = gym.make('Pendulum-v0')
        env.seed(SEED)
        envs.append(env)

    net = A3CNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    agent = A3CAgent(net, device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count= STEPS)

    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0
    epoch = 0

    print_time = time()
    start_time = datetime.now()
    train_ep = 0
    total_rewards = []
    batch = []
    for idx, exp in enumerate(exp_source):
        batch.append(exp)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            train_ep += 1
            total_rewards.append(new_reward[0])
            mean = np.mean(total_rewards[-100:])
            if time()-print_time > 1:
                print(f'epoch:{epoch:6} mean:{mean:7.2f}, policy_loss:{policy_loss:7.2f},value_loss:{value_loss:7.2f}, entropy_loss:{entropy_loss:7.2f}, reward:{new_reward[0]:7.2f}')
                print_time = time()
            if mean > SOLVE_BOUND:
                duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                if args.save: torch.save(net.state_dict(),'cont_lunar_a3c.dat')
                # if args.play: play(env,agent)
                break
        if train_ep < TRAIN_EPISODES:
            continue
        train_ep = 0
        epoch += 1
        states_v, actions_v, q_refs_v = unpack_batch_a2c(batch, net, GAMMA**STEPS)
        batch.clear()

        optimizer.zero_grad()
        mu_v, var_v, val_v = net(states_v)
        # Value loss:
        value_loss = F.mse_loss(val_v.squeeze(-1), q_refs_v)
        # Policy loss:
        adv_v = q_refs_v.unsqueeze(-1) - val_v.detach()
        log_prob_v = adv_v * calc_log_prob(mu_v, var_v, actions_v)
        policy_loss = - log_prob_v.mean()
        # Entropy loss:
        ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
        entropy_loss = ENTROPY_BETA * ent_v.mean()
        loss = policy_loss + entropy_loss + value_loss
        loss.backward()
        nn_utils.clip_grad_norm_(net.parameters(),CLIP_GRAD)
        optimizer.step()
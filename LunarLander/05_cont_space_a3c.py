'''
The worst performance ever! 
Rewards kept decreasing into negative territory no matter how long you trian.

**** Not worth the training ****
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

TotalRewards = namedtuple('TotalRewards','reward')

GAMMA = 0.99
STEPS = 10
ENV_ID = 'LunarLanderContinuous-v2'
ENTROPY_BETA = 0.02
CLIP_GRAD = 0.1
NUM_PROCESSES = mp.cpu_count()
ENVS = 2
MINI_BATCH_SIZE = 128
BATCH_SIZE = 2048
SOLVE_BOUND = 150
LR = 1e-3
TEST_ITER = 200


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
        actions = np.clip(rand_a,-1,1)
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


def data_func(net, device, train_queue):
    envs = [gym.make(ENV_ID) for _ in range(ENVS)]
    agent = A3CAgent(net, device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count= STEPS)
    mini_batch = []
    for exp in exp_source:
        new_reward = exp_source.pop_rewards_steps()
        if new_reward:
            data = TotalRewards(np.mean(new_reward))
            train_queue.put(data)
            continue #Not sure I need this 
        mini_batch.append(exp)
        if len(mini_batch) < MINI_BATCH_SIZE:
            continue
        data = unpack_batch(mini_batch, net, GAMMA, STEPS, device=device)
        train_queue.put(data)
        mini_batch.clear()


def calc_log_prob(mu_v,var_v,action_v):
    p1 = -((action_v - mu_v)**2)/(2*var_v.clamp(min=1e-3))
    p2 = - torch.log2(torch.sqrt(2*math.pi*var_v))
    return p1+p2

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
            action = np.clip(action, -1, 1)
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
    mp.set_start_method('spawn')
    os.environ['MPS_NUM_THREADS']='1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='Play and episode when training is complete')
    parser.add_argument('--save', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_a3c.dat"')
    parser.add_argument('--steps','-s', default=10, type=int, help='Steps used to discount reward') # This is new in Policy Gradient Method
    args = parser.parse_args()
     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('LunarLanderContinuous-v2')
    net = A3CNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net.share_memory()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    train_queue = mp.Queue(maxsize=NUM_PROCESSES)
    procs = []
    for i in range(NUM_PROCESSES):
        child_proc = mp.Process(target=data_func, args=(net,device,train_queue),name=str(i))
        child_proc.start()
        procs.append(child_proc)
    
    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0
    epoch = 0
    idx_iter = 0
    total_rewards = []
    batch = []
    print_time = time()
    start_time = datetime.now()
    batch_size = 0
    states,actions,q_refs = [],[],[]
    try:
        while True:
            idx_iter += 1
            train = train_queue.get()
            if isinstance(train, TotalRewards):
                total_rewards.append(train.reward)
                mean = np.mean(total_rewards[-100:])
                if time()-print_time > 1:
                    print(f'epoch:{epoch:6} mean:{mean:7.2f}, policy_loss:{policy_loss:7.2f},value_loss:{value_loss:7.2f}, entropy_loss:{entropy_loss:7.2f}, reward:{train.reward:7.2f}')
                    print_time = time()
                if mean > SOLVE_BOUND:
                    duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                    print(f'Solved in {duration}')
                    if args.save: torch.save(net.state_dict(),'cont_lunar_a3c.dat')
                    if args.play: play(env,agent)
                    break
                continue
            mini_states,mini_actions,mini_q_refs = train
            states.append(mini_states)
            actions.append(mini_actions)
            q_refs.append(mini_q_refs)
            batch_size += len(mini_actions)
            if batch_size < BATCH_SIZE:
                continue
            batch_size = 0
            epoch += 1
            states_v = torch.cat(states)
            actions_v = torch.cat(actions)
            q_refs_v = torch.cat(q_refs)
            states.clear()
            actions.clear()
            q_refs.clear()

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
    finally:
        for p in procs:
            p.terminate()
            p.join()
'''
epoch: 23541 mean: 139.08, policy_loss:  -0.06, value_loss:   0.72, entropy_loss:  -0.01, reward  49.30
epoch: 23551 mean: 139.23, policy_loss:  -0.09, value_loss:  13.88, entropy_loss:  -0.01, reward 258.64
epoch: 23562 mean: 147.31, policy_loss:   0.02, value_loss:  15.12, entropy_loss:  -0.01, reward 285.85
Solved in 0:57:55


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.multiprocessing as mp

import os
import gym
import ptan
import argparse
import numpy as np
from time import time
from collections import namedtuple
from datetime import datetime, timedelta


STEPS = 1
GAMMA = 0.99
MINI_BATCH_SIZE = 64
BATCH_SIZE = 512
ENVS = 2
PROCESS_COUNT = mp.cpu_count()
ENV_ID = 'LunarLander-v2'
SOLVE_BOUND = 150
LR = 1e-3
ENTROPY_BETA = 0.02
CLIP_GRAD = 0.1

class A3CNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, 256),
        nn.ReLU())
        self.policy = nn.Linear(256, act_size)
        self.value = nn.Linear(256, 1)
    
    def forward(self, x):
        base = self.base(x)
        return self.policy(base), self.value(base)


TotalRewards = namedtuple('TotalRewards','reward')

def unpack_batch(batch,net,device='cpu'):
    s,a,r,ls = [],[],[],[]
    not_done_idx = []
    for idx,exp in enumerate(batch):
        s.append(np.array(exp.state, copy=False))
        a.append(int(exp.action))
        r.append(np.array(exp.reward, dtype=np.float32))
        if exp.last_state is not None:
            ls.append(np.array(exp.last_state, copy=False))
            not_done_idx.append(idx)
    r_np = np.array(r, copy=False)
    if not_done_idx:
        ls_v = torch.FloatTensor(np.array(ls, copy=False)).to(device)
        q_val_np = net(ls_v)[1].data.cpu().numpy()[:,0]
        q_val_np *= GAMMA**STEPS
        r_np[not_done_idx] += q_val_np
    ref_value = torch.tensor(r_np)
    states = torch.FloatTensor(np.array(s, copy=False))
    actions = torch.LongTensor(a)
    return states, actions, ref_value


def data_func(net,device,train_queue):
    envs = [gym.make(ENV_ID) for _ in range(ENVS)]
    agent = ptan.agent.ActorCriticAgent(net, device=device, apply_softmax=True,
                                         preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=STEPS)
    mini_batch = []
    for exp in exp_source:
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            data = TotalRewards(reward=np.mean(new_reward))
            train_queue.put(data)
        mini_batch.append(exp)

        if len(mini_batch) < MINI_BATCH_SIZE:
            continue
        data = unpack_batch(mini_batch,net,device)
        train_queue.put(data)
        mini_batch.clear()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='Play and episode when training is complete')
    parser.add_argument('--save', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_a3c.dat"')
    parser.add_argument('--steps','-s', default=10, type=int, help='Steps used to discount reward') # This is new in Policy Gradient Method
    args = parser.parse_args()

    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS']='1'

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(ENV_ID)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    net = A3CNet(obs_size, act_size).to(device)
    net.share_memory()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    train_queue = mp.Queue(maxsize=PROCESS_COUNT)

    multi_procs = []
    for i in range(PROCESS_COUNT):
        child_proc = mp.Process(target=data_func,args=(net,device,train_queue),name=str(i))
        child_proc.start()
        multi_procs.append(child_proc)

    epoch = 0
    value_loss=0.0
    policy_loss = 0.0
    entropy_loss = 0.0
    batch_size = 0
    total_rewards = []
    print_time = time()
    start_time = datetime.now()
    batch_states = []
    batch_actions = []
    batch_ref_vals = []
    try:
        while True:
            train_data = train_queue.get()
            if isinstance(train_data, TotalRewards):
                total_rewards.append(train_data.reward)
                mean = np.mean(total_rewards[-100:])
                if time() - print_time > 1:
                    print(f'epoch:{epoch:6} mean:{mean:7.2f}, policy_loss:{policy_loss:7.2f}, value_loss:{value_loss:7.2f}, entropy_loss:{entropy_loss:7.2f}, reward{train_data.reward:7.2f}')
                    print_time = time()
                if mean > SOLVE_BOUND:
                    duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                    print(f'Solved in {duration}')
                    if args.save: torch.save(net.state_dict(),'lunar_a3c.dat')
                    if args.play: play(env,agent)
                    break
                continue
            mini_states, mini_actions, mini_ref_vals = train_data
            batch_states.append(mini_states)
            batch_actions.append(mini_actions)
            batch_ref_vals.append(mini_ref_vals)
            batch_size += len(mini_states)
            if batch_size < BATCH_SIZE:
                continue
            
            states = torch.cat(batch_states).to(device)
            actions = torch.cat(batch_actions)
            ref_values = torch.cat(batch_ref_vals)
            batch_states.clear()
            batch_actions.clear()
            batch_ref_vals.clear()
            batch_size = 0
            del mini_states, mini_actions, mini_ref_vals
            epoch += 1
            optimizer.zero_grad()
            policy, values = net(states)
            value_loss = F.mse_loss(values.squeeze(-1), ref_values)
            
            log_prob = F.log_softmax(policy, dim=1)
            log_prob_a = log_prob[range(len(actions)),actions]
            adv = ref_values - values.detach()[:,0]
            policy_loss = -(log_prob_a * adv).mean()

            prob = F.softmax(policy, dim=1)
            ent = (prob * log_prob).sum(dim=1).mean()
            entropy_loss = ENTROPY_BETA * ent

            loss = policy_loss + value_loss + entropy_loss
            loss.backward()
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
    finally:
        for p in multi_procs:
            p.terminate()
            p.join()


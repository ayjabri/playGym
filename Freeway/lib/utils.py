#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:26:36 2020

@author: Ayman Al Jabri
"""
import numpy as np
import torch
import torch.nn.functional as F



def unpack_batch(batch):
    s,a,r,d,l = [],[],[],[],[]
    for exp in batch:
        s.append(np.array(exp.state, copy=False))
        a.append(exp.action)
        r.append(exp.reward)
        d.append(exp.last_state is None)
        if exp.last_state is None:
            l.append(np.array(exp.state, copy=False))
        else:
            l.append(np.array(exp.last_state, copy=False))
    return (np.array(s, copy=False),
            np.array(a),
            np.array(r,dtype=np.float32),
            np.array(d),
            np.array(l,copy=False))


def calc_dqn_loss(batch, net, tgt_net, gamma, device='cpu'):
    states,actions,rewards,dones,last_states = unpack_batch(batch)
    
    states_v=torch.FloatTensor(states).to(device)
    last_states_v=torch.FloatTensor(last_states).to(device)
    rewards_v = torch.tensor(rewards)
    
    size = len(actions)
    qval_v = net(states_v)
    qval_a = qval_v[range(size),actions]
    with torch.no_grad():
        next_qval = tgt_net.target_model(last_states_v)
        best_next_qval = next_qval.max(dim=1)[0]
        best_next_qval[dones] = 0
    future_rewards = rewards_v + gamma * best_next_qval
    
    return F.mse_loss(qval_a, future_rewards)
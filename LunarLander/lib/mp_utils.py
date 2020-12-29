# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import ptan
from collections import namedtuple

EpisodeEnd = namedtuple('EpisodeEnd',['step','reward','epsilon'])

class Net(nn.Module):
    def __init__(self, shape, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(shape, 256),
                                   nn.ReLU()                                   
                                   )
        self.val = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_actions)

        
    def forward(self, x):
        x = self.layer(x)
        val = self.val(x)
        adv = self.adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


def unpack_batch(batch):
    s,a,r,d,l = [],[],[],[],[]
    for exp in batch:
        s.append(exp.state)
        a.append(exp.action)
        r.append(exp.reward)
        d.append(exp.last_state is None)
        if exp.last_state is None:
            l.append(exp.state)
        else:
            l.append(exp.last_state)
    return (np.array(s),
            np.array(a),
            np.array(r),
            np.array(d),
            np.array(l))


def calc_loss(batch, net, tgt_net, gamma):
    states, actions, rewards, dones, last_states = unpack_batch(batch)

    s_v = torch.FloatTensor(states)
    a_v = torch.LongTensor(actions)
    r_v = torch.FloatTensor(rewards)
    d_v = torch.BoolTensor(dones)
    l_v = torch.FloatTensor(last_states)

    q_a_v = net(s_v).gather(dim=1, index=a_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        max_qp_s_a = tgt_net.target_model(l_v).max(dim=1)[0]
        max_qp_s_a[d_v] = 0.0
        y = r_v + gamma * max_qp_s_a
    return F.mse_loss(q_a_v, y)


def data_fun(net,exp_queue,ENV_ID,STEPS=1):
    """
    Stores ptan FirstLast experiences in a multiprocess Queue()

    Parameters
    ----------
    net : Deep-Q Neural Netwok class
        Can be any DQN. Tested with DuelDQN network
        
    exp_queue : Pytorch Multiprocessing.Queue()
        Shared Queue to store experiences.
        
    ENV_ID : Int
        Name of Gym OpenAI game.
        
    STEPS : Int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    Stores experiences in a multiprocessing Queue(). It also stores step,reward and epsilon
    as named tuple (EndEpisode) at the end of each episode.
    
    Use as target for Multiprocessing.

    """
    env0 = gym.make(ENV_ID)
    selector0 = ptan.actions.EpsilonGreedyActionSelector()
    agent0 = ptan.agent.DQNAgent(net, selector0,preprocessor=ptan.agent.float32_preprocessor)
    eps_tracker0 = ptan.actions.EpsilonTracker(selector0, 1.0, 0.02, 10_000)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env0,agent0,0.99,steps_count=STEPS)
    step = 0
    for exp in exp_source:
        step += 1
        eps_tracker0.frame(step)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(step, new_reward[0],selector0.epsilon))
        exp_queue.put(exp)        



class MPBatchGenerator:
    """
    Yields batchs from experiences stored in multiprocess Queue()
    Parameters:
    -------
    buffer: ptan.experience.ExperienceReplayBuffer(exp_source=None)
    
    exp_queue: Torch Multiprocessing Queue()
    
    batch_size: int
        The size of batch to generate
    
    BATCH_MULT: int. Default to 1
        Multiply batch size by this number
    """
    def __init__(self,buffer,exp_queue,initial,batch_size,multiplier):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.initial = initial
        self.batch_size = batch_size
        self.multiplier = multiplier
        self._total_rewards = []
        self.frame = 0
        self.episode = 0
        self.epsilon = 0.0
    def pop_rewards_idx_eps(self):
        res = list(self._total_rewards)
        self._total_rewards.clear()
        return res
    def __len__(self):
        return len(self.buffer)
    def __iter__(self):
        while True:
            while not self.exp_queue.empty():
                exp = self.exp_queue.get()
                if isinstance(exp, EpisodeEnd):
                    self._total_rewards.append(exp.reward)#(exp.reward,exp.step,exp.epsilon)
                    self.frame = exp.step
                    self.epsilon = exp.epsilon
                    self.episode += 1
                else:
                    self.buffer._add(exp)
                    self.frame += 1
            if len(self.buffer)<self.initial:
                continue
            yield self.buffer.sample(self.batch_size * self.multiplier)

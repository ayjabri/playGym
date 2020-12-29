#!/usr/bin/env python3
import gym
import os
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 1280

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 2
MICRO_BATCH_SIZE = 128


# ENV_NAME = "CartPole-v0"
# REWARD_BOUND = 195

ENV_NAME = "LunarLander-v2"
REWARD_BOUND = 50


def make_env():
    return gym.make(ENV_NAME)


TotalReward = collections.namedtuple('TotalReward', field_names='reward')


def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    micro_batch = []

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            data = TotalReward(reward=np.mean(new_rewards))
            train_queue.put(data)

        micro_batch.append(exp)
        if len(micro_batch) < MICRO_BATCH_SIZE:
            continue

        data = common.unpack_a3c_batch(
            micro_batch, net, device=device,
            last_val_gamma=GAMMA ** REWARD_STEPS)
        train_queue.put(data)
        micro_batch.clear()



class A3CNet(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()

        self.input = nn.Linear(in_features[0], 256)
        self.fc_policy = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)

    def forward(self, x):
        base = F.relu(self.input(x.float()))
        return self.fc_policy(base), self.fc_val(base)



if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    writer = SummaryWriter(comment=f"-a3c-data_lunar_{args.name}")

    env = make_env()
    net = A3CNet(env.observation_space.shape,
                          env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,
                           eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func,
                               args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch_states = []
    batch_actions = []
    batch_vals_ref = []
    step_idx = 0
    batch_size = 0

    try:
        with common.RewardTracker(writer, REWARD_BOUND) as tracker:
            with ptan.common.utils.TBMeanTracker(
                    writer, 100) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward,
                                          step_idx):
                            break
                        continue

                    states_t, actions_t, vals_ref_t = train_entry
                    batch_states.append(states_t)
                    batch_actions.append(actions_t)
                    batch_vals_ref.append(vals_ref_t)
                    step_idx += states_t.size()[0]
                    batch_size += states_t.size()[0]
                    if batch_size < BATCH_SIZE:
                        continue

                    states_v = torch.cat(batch_states)
                    actions_t = torch.cat(batch_actions)
                    vals_ref_v = torch.cat(batch_vals_ref)
                    batch_states.clear()
                    batch_actions.clear()
                    batch_vals_ref.clear()
                    batch_size = 0

                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)

                    loss_value_v = F.mse_loss(
                        value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.detach()
                    size = states_v.size()[0]
                    log_p_a = log_prob_v[range(size), actions_t]
                    log_prob_actions_v = adv_v * log_p_a
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    ent = (prob_v * log_prob_v).sum(dim=1).mean()
                    entropy_loss_v = ENTROPY_BETA * ent

                    loss_v = loss_policy_v #+entropy_loss_v + loss_value_v + \
                             
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(
                        net.parameters(), CLIP_GRAD)
                    optimizer.step()

                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v,
                                     step_idx)
                    tb_tracker.track("loss_entropy",
                                     entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy",
                                     loss_policy_v, step_idx)
                    tb_tracker.track("loss_value",
                                     loss_value_v, step_idx)
                    tb_tracker.track("loss_total",
                                     loss_v, step_idx)
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

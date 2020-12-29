"""
Solving Lunar Lander using Deep-Q Learning Method
Network: Duel Network with one hidden layer and two heads (Value and Action_Advantage)
Loss : Bellman equation loss= Reward + Gamma x Q(s,a) vs Max(Q(s`,a)) using the target network

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import ptan
import argparse
from time import time
from datetime import datetime, timedelta


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



@torch.no_grad()
def play(env, agent):
    state = env.reset()
    rewards= 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


LR = 1e-3
SOLVE = 150
MIN = 2000
BATCH_SIZE = 256


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_dqn.dat"')
    parser.add_argument('--steps', default=1, type=int, help='Number of steps to skip when training')
    parser.add_argument('--gamma', default=0.99, type=int, help='Discount Rate')
    args = parser.parse_args()
    STEPS = args.steps
    GAMMA = args.gamma

    gamma = GAMMA**STEPS

    env = gym.make('LunarLander-v2')
    net = Net(env.observation_space.shape[0],env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, 1.0, 0.02, 20_000)
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, 150_000)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)



    total_rewards = []
    frame_idx = 0
    epoch = 0
    loss = 0.0
    start = time()
    print(net)
    start_time = datetime.now()
    while True:
        frame_idx += 1
        eps_tracker.frame(frame_idx)
        buffer.populate(1)

        reward = exp_source.pop_total_rewards()
        if reward:
            epoch += 1
            total_rewards.append(reward[0])
            mean = np.mean(total_rewards[-100:])
            if time()-start >1:
                print(f'epoch:{epoch:6} mean:{mean:7.2f}, loss:{loss:7.2f}, reward{reward[0]:7.2f} epsilon:{selector.epsilon:4.2f}')
                start = time()
            if mean > SOLVE:
                duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                if args.save: torch.save(net.state_dict(),'lunar_dqn.dat')
                if args.play: play(env,agent)
                break
        
        if len(buffer) < MIN:
            continue
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, net, tgt_net, gamma)
        loss.backward()
        optimizer.step()

        if frame_idx % 1000 == 0:
            tgt_net.sync()

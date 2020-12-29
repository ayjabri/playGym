"""
        *********  Using Multipocessing ***********
Solving Lunar Lander using Deep-Q Learning Method
Network: Duel Network with one hidden layer and two heads (Value and Action_Advantage)
Loss : Bellman equation loss= Reward + Gamma x Q(s,a) vs Max(Q(s`,a)) using the target network

Results
-------
I was able to solve this in 9 minutes (dead), which is 2 minutes faster than the exact
same configurations without the multiprocess.

Another run using N_MP as multiplier (i.e. increase batch size):
    Solved in 0:07:45
"""
import torch
import torch.multiprocessing as mp

import os
import gym
import ptan
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from lib import mp_utils
from collections import deque


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

if __name__=='__main__':
    N_MP = 4
    LR = 1e-3
    SOLVE = 195
    INIT_REPLAY = 2000
    BATCH_SIZE = 256
    GAMMA = 0.99
    STEPS = 1
    ENV_ID = 'LunarLander-v2'
    mp.set_start_method('fork',force=True)
    os.environ['OMP_NUM_THREADS'] = "4"
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',default=False, help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=True, help='Save a copy of the trained network in current directory as "lunar_dqn.dat"')
    parser.add_argument('--steps', default=STEPS, type=int, help='Number of steps to skip when training')
    parser.add_argument('--gamma', default=GAMMA, type=int, help='Discount Rate')
    args = parser.parse_args()
    STEPS = args.steps
    GAMMA = args.gamma

    gamma = GAMMA**STEPS

    env = gym.make(ENV_ID)
    net = mp_utils.Net(env.observation_space.shape[0],env.action_space.n)
    net.share_memory()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    buffer = ptan.experience.ExperienceReplayBuffer(None, 150_000)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    exp_queue = mp.Queue(N_MP*4)
    proc = mp.Process(target=mp_utils.data_fun,args=(net,exp_queue, ENV_ID, STEPS))
    proc.start()
    generator = mp_utils.MPBatchGenerator(buffer,exp_queue,INIT_REPLAY,BATCH_SIZE,N_MP) #
    
    print(net)
    pt = time()
    loss = 0.0
    start_time = datetime.now()
    total_rewards = deque(maxlen=100)
    epoch = 0
    for batch in generator:
        new_reward = generator.pop_rewards_idx_eps()   
        if new_reward:
            total_rewards.extend(new_reward)
            mean = np.mean(total_rewards)
            if mean > SOLVE:
                duration = timedelta(seconds = (datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                if args.save: torch.save(net.state_dict(),'lunar_dqn_mp.dat')
                if args.play: play(env,agent)
                break
            if time()-pt >1:
                print(f'epoch:{epoch:6} mean:{mean:7.2f}, loss:{loss:7.2f}, reward{new_reward[0]:7.2f} epsilon:{generator.epsilon:4.2f}')
                pt = time()

        optimizer.zero_grad()
        loss = mp_utils.calc_loss(batch, net, tgt_net, gamma)
        loss.backward()
        optimizer.step()
        epoch += 1
        if generator.frame % 1000 == 0:
            tgt_net.sync()
        del batch
    proc.terminate()
    proc.join()
    exp_queue.close()
    exp_queue.join_thread()

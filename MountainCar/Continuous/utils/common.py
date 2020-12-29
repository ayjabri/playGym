# -*- coding: utf-8 -*-

import gym
import ptan
import torch
import numpy as np
import argparse


@torch.no_grad()
def play_continuous(env, model, device='cpu'):
    rewards = 0.0
    steps = 0
    obs = env.reset()
    while True:
        env.render()
        obs_v = ptan.agent.float32_preprocessor(obs)
        mu_v = model(obs_v)[0]
        mu = mu_v.data.cpu().numpy()
        action = np.clip(mu, -1,1)
        obs, r, done, _ = env.step(action)
        rewards += r
        steps += 1
        if done:
            print(f'finished in {steps} steps with {rewards:.2f} rewards')
            break
    env.close()
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', help='Record video in place if selected')
    args = parser.parse_args()

    if args.record:
        env = gym.wrappers.Monitor(env, 'recordings/')

    play_continuous(env, net)
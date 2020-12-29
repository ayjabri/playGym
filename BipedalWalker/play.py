import argparse
import gym

import ptan
import numpy as np
import torch
import torch.nn as nn

ENV_ID = "BipedalWalker-v2"


@torch.no_grad()
def play(env,act_net, render=False):
    obs = env.reset()
    rewards = 0.0
    while True:
        if render: env.render()
        obs_v = ptan.agent.float32_preprocessor(obs)
        action = act_net(obs_v).data.numpy()
        obs,r,done,_=env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()
    


class DDPGActorNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, 400),
                                  nn.ReLU(),
                                  nn.Linear(400, 300),
                                  nn.ReLU(),
                                  nn.Linear(300, act_size),
                                  nn.Tanh(),
                                  )
    def forward(self, x):
        return self.base(x)
    

PATH ='playGym/BipedalWalker/' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action ='store_true',default=True, help="Show while playing")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-f", "--fname", default=PATH, help="path + file_name of the trained network (state_dict)")
    args = parser.parse_args()

    PATH = args.fname
    

    env = gym.make(args.env)
    
    if args.record:
        env = gym.wrappers.Monitor(env, args.record,force=True, resume=True)

    net = DDPGActorNet(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(PATH, map_location='cpu'))
    play(env, net, args.render)



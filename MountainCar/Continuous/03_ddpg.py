'''
Deep Deterministic Policy Gradient (DDPG) method is from the Actor-Critic family
but it is slightly different in the way it uses Critic network.
In A2C the critic is used to get a baseline for our discounted rewards, while in 
DDPG it returns Q(s,a) value which represents the total discounted rewards
of action "a" in state "s".
The policy is deterministic, which means the actor returns the action directly from 
Actor network. Unlike A2C which is stochastic i.e. returns probability distribution
parameters of the action (mean and variance).
'''
import gym
import ptan
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from time import time
from datetime import datetime

# Actor Network
class DDPGActor(nn.Module):
    '''
    Actor Network: what we want from the actor is to spit out
         actions when fed the states
    Tanh: activation squeezes all values between -1,1 to match the
        continuous action space range of the environment.
    '''
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(obs_size, 400),
                                    nn.ReLU(),
                                    nn.Linear(400, 300),
                                    nn.ReLU(),
                                    nn.Linear(300, act_size),
                                    nn.Tanh(),
                                    )
    
    def forward(self,x):
        return self.layer(x)


# Critic Network
class DDPGCritic(nn.Module):
    '''
    Critic Network: the role of critic network is to return Q(s,a)
        which is the discounted rewards of action "a" in state "s".
        It's then used to evaluate the goodness of the actions, by 
        maximizing Q(s,a)
        The architecture is simple: we input states and actions, and get
        a single value
    '''
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.net_in = nn.Sequential(nn.Linear(obs_size, 400),
                                    nn.ReLU(),
                                    )
        self.q_s_a = nn.Sequential(nn.Linear(400 + act_size, 300),
                                    nn.ReLU(),
                                    nn.Linear(300, 1),)
    
    def forward(self,s, a):
        net_out = self.net_in(s)
        return self.q_s_a(torch.cat([net_out, a], dim=1))


class  DDPGAgent(ptan.agent.BaseAgent):
    '''
    DDPG Agent: returns noisy actions by adding a noise from a normally
        distributed values (mu=0,std=1) scaled by epsilon
    '''
    def __init__(self, model, low=-1, high=1, device='cpu', epsilon=0.1):
        self.model = model
        self.device = device
        self.low = low
        self.high = high
        self.epsilon = epsilon
    
    def __call__(self, state, agent_states):
        state_v = ptan.agent.float32_preprocessor(state).to(self.device)
        mu_np = self.model(state_v).data.cpu().numpy()
        mu_np += np.random.normal(size= mu_np.shape) * self.epsilon
        actions = np.clip(mu_np, self.low, self.high)
        return actions, agent_states


def unpack_batch_dqn(batch, device='cpu'):
    '''
    Unpack a batch of experiences returning
    States: -> float32
    Actions: -> float32
    Rewards: -> float32
    Dones: -> bool
    Last_states: -> float32
    '''
    states,actions,rewards,dones,last_states=[],[],[],[],[]
    for exp in batch:
        states.append(np.array(exp.state, copy=False))
        actions.append(np.array(exp.action, copy=False))
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(np.array(exp.state, copy=False))
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return (torch.FloatTensor(np.array(states, copy=False)).to(device),
            torch.FloatTensor(np.array(actions, copy=False)).to(device),
            torch.tensor(np.array(rewards, dtype=np.float32)).to(device),
            torch.BoolTensor(dones).to(device),
            torch.FloatTensor(np.array(last_states, copy=False)).to(device))


class RewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        if self.state[0] >= 0.5:
            reward = 1000
        elif self.state[0] > -0.7:
            reward = abs(self.state[0])
        elif self.state[0] < -1.19:
            reward = -10
        return reward

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Enable Cuda')
    args = parser.parse_args()

    ENV_ID = 'MountainCarContinuous-v0'
    LR_ACTOR = 5e-5
    LR_CRITIC = 1e-3
    BATCH_SIZE = 64
    GAMMA = 0.99
    STEPS  = 1
    SOVLE_BOUND = 1400
    BUF_SIZE = 100_000
    INIT_REPLAY = 10_000

    device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    print(f'Device is {device}')

    env = RewardWrapper(gym.make(ENV_ID))
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    act_net = DDPGActor(obs_size, act_size).to(device)
    crt_net = DDPGCritic(obs_size, act_size).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    print(act_net)
    print(crt_net)

    agent = DDPGAgent(act_net, device=device, epsilon=0.3)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, BUF_SIZE)

    act_optim = torch.optim.Adam(act_net.parameters(), lr= LR_ACTOR)
    crt_optim = torch.optim.Adam(crt_net.parameters(), lr= LR_CRITIC)

    total_rewards = deque(maxlen=100)
    start_time = datetime.now()
    print_time = time()
    frame_idx = 0
    frame_speed = 0
    episode = 0
    loss = 0.0
    best_reward = None
    while True:
        frame_idx += 1
        buffer.populate(1)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            episode += 1
            total_rewards.append(new_reward[0])
            mean = np.mean(total_rewards)
            if mean > SOVLE_BOUND:
                print(f'Solved after reaching {mean:.2f} rewards within {datetime.now()-start_time}!')
                fname = ENV_ID + "_.dat"
                torch.save(act_net.state_dict(),fname)
                break
            if time()-print_time > 2:
                fps = (frame_idx - frame_speed)/(time()-print_time)
                print(f'{frame_idx:7}: episode:{episode:6}, mean:{mean:7.2f}, loss:{loss:7.2f}, speed: {fps:6.2f} fps')
                print_time = time()
                frame_speed = frame_idx
        if len(buffer) < INIT_REPLAY:
            continue

        
        batch = buffer.sample(BATCH_SIZE)
        states, actions, rewards, dones, last_states =\
            unpack_batch_dqn(batch, device)
        
        # train critic
        crt_optim.zero_grad()
        q_sa = crt_net(states, actions)
        last_a = tgt_act_net.target_model(last_states)
        q_sa_p = tgt_crt_net.target_model(last_states, last_a)
        q_sa_p[dones] = 0.0
        # apply Bellman equation
        q_ref_v = rewards.unsqueeze(dim=-1) + q_sa_p * GAMMA
        critic_loss = F.mse_loss(q_sa, q_ref_v.detach())
        critic_loss.backward()
        crt_optim.step()

        # train actor
        act_optim.zero_grad()
        cur_actions = act_net(states)
        actor_loss = -crt_net(states, cur_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        act_optim.step()

        loss = actor_loss.item()

        tgt_crt_net.alpha_sync(alpha = 1 - 1e-3)
        tgt_act_net.alpha_sync(alpha = 1 - 1e-3)

        # if frame_idx % ITER_TEST ==0:

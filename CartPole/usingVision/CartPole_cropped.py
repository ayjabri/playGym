# -*- coding: utf-8 -*-
"""
Cart Pole using Conv1d network

"""

import time
import gym
import torch
import numpy as np
import random
import matplotlib
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
from collections import deque

#%% Setup Display

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


#%% Deep Q Network

class DQN(nn.Module):
    def __init__(self,img_height):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=img_height,out_channels=64,kernel_size=7,bias=False)
        self.norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=7,bias=False)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=7,bias=False)
        self.norm3 = nn.BatchNorm1d(128)
        self.adapt = nn.AdaptiveMaxPool1d(1)
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(in_features=128,out_features=32)
        self.bn1   = nn.BatchNorm1d(32)
        self.fc2   = nn.Linear(in_features=32,out_features=2)
    def forward(self,s):
        s = F.relu(self.norm1(self.conv1(s)))
        s = F.relu(self.norm2(self.conv2(s)))
        s = F.relu(self.norm3(self.conv3(s)))
        s = self.adapt(s)
        s = self.flat(s)
        s = F.relu((self.fc1(s)))
        return self.fc2(s)

#%% Replay Memory
Experience = namedtuple('Experience',('state','action','next_state','reward'))

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        
    def __len__(self):
        return len(self.memory)
    
    def store(self,experience):
        self.memory.append(experience)
    
    def sample(self,batch_size):
        batch_size = min(batch_size,len(self.memory))
        return random.sample(self.memory,batch_size)
    
    def can_provide_sample(self,batch_size):
        return batch_size <= len(self.memory)
    
#%% Strategy

class EpsilonGreedyStrategy(object):
    def __init__(self,start,end,decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self,current_step):
        return self.end + (self.start - self.end)*np.exp(-1 * self.decay * current_step)

#%% Agent

class Agent(object):
    def __init__(self,strategy,num_actions,device):
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.explore = True
        self.device = device
        
    def select_action(self,state,policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        if rate > random.random():
            self.explore = True
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # Explore
        with torch.no_grad():
            self.explore = False
            return policy_net(state).argmax(dim=1).to(self.device) # Exploite

#%% Environment Manger

class CartPoleEnvManager(object):
    def __init__(self,device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.world_width = self.env.x_threshold * 2
    
    def reset(self):
        self.env.reset()
        self.current_screen = None
        self.done = False
    
    def close(self):
        self.env.close()
    
    def render(self,mode='human'):
        return self.env.render(mode)
    
    def num_available_actions(self):
        return self.env.action_space.n
    
    def take_action(self,action):
        _,reward,self.done,_ = self.env.step(action.item())
        return torch.tensor([reward],device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_cart_location(self,screen):
        loc = self.env.state[0]
        scale = screen.shape[2] / self.world_width
        return int(loc * scale + screen.shape[2] / 2)
        
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2-s1
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_processed_screen(self):
        screen = self.env.render('rgb_array').transpose(2,0,1)
        cart_loc = self.get_cart_location(screen)
        screen = self.crop_screen(screen,cart_loc)
        return self.transform_screen_data(screen)
    
    def crop_screen(self,screen,cart_loc):
        _,screen_height,screen_width = screen.shape
        delta = 200
        # Strip off the top and buttom of the screen
        top = int(0.4 * screen_height)
        buttom = int(0.8 * screen_height)
        if cart_loc > (screen_width - delta):
            screen = screen[:,top:buttom,screen_width-2*delta:]
        elif cart_loc < delta:
            screen = screen[:,top:buttom,:2*delta]
        else:
            screen = screen[:,top:buttom,cart_loc-delta:cart_loc+delta]
        return screen
    
    def transform_screen_data(self,screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32)/255
        screen = torch.from_numpy(screen)
        
        self.transforms = T.Compose([T.ToPILImage(),
                                     T.Grayscale(),
                                    T.Resize(40),
                                    T.ToTensor()])
        
        return self.transforms(screen).to(self.device)

#%% Plot Training

def plot(values,moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training .....')
    plt.xlabel('Episodes')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_avg(moving_avg_period,values))
    plt.pause(0.001)
    if is_ipython: display.clear_output(wait=True)

def get_moving_avg(period,values):
    values = torch.tensor(values,dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(0,period,1).mean(dim=1).flatten(0)
        moving_avg = torch.cat([torch.zeros(period),moving_avg])
    else:
        moving_avg = torch.zeros(len(values))
    return moving_avg.numpy()

#%% Initiate Parameters & Objects

batch_size = 10
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100_000
lr = 0.001
num_episodes = 100

device = ('cuda' if torch.cuda.is_available() else 'cpu')

em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start,eps_end,eps_decay)
agent = Agent(strategy,em.num_available_actions(),device)
memory = ReplayMemory(memory_size)

policy_net = DQN(40).to(device)
target_net = DQN(40).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(),lr=lr)
em.close()

#%% Utilities

def extract_tensors(experiences):
    # converts batch of experiences to an experiences of batchs
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1,t2,t3,t4)

class QValues():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1,index=actions.unsqueeze(-1))
    
    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

#%% Training .....

episode_durations = []

for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    for timestep in count():
        action = agent.select_action(state,policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.store(Experience(state,action,next_state,reward))
        state = next_state
        
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states,actions,rewards,next_states = extract_tensors(experiences)
            
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            
            loss = F.mse_loss(current_q_values,target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations,100)
            break
    if episode % target_update ==0:
        target_net.load_state_dict(policy_net.state_dict())
        
    em.close()


#%% Play
values = []
all_actions = {}

for e in range(100):
    rewards = 0
    actions = []
    em.reset()
    state = em.get_state()
    for step in count():
        action = target_net(state).argmax(1)
        actions.append(action.item())
        reward = em.take_action(action)
        next_state = em.get_state()
        state = next_state
        rewards += reward
        if em.done:
            values.append(rewards)
            all_actions[e] = actions
            plot(values,5)
            break
em.close()
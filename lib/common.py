import torch
import torch.nn.functional as F
import time
import sys
import numpy as np
from types import SimpleNamespace

HYPERPARAMS = SimpleNamespace(**{
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'breakout-small': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    }),
    'breakout': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'invaders': SimpleNamespace(**{
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    }),
    'cartpole': SimpleNamespace(**{
        'env_name': "CartPole-v0",
        'stop_reward': 180.0,
        'run_name': 'CartPole',
        'replay_size': 1000,
        'replay_initial': 64,
        'target_net_sync': 10,
        'epsilon_frames': 5000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 1e-2,
        'gamma': 0.95,
        'batch_size': 32
    }),
        'mountaincar': SimpleNamespace(**{
        'env_name': "MountainCar-v0",
        'stop_reward': -76.5,
        'run_name': 'CartPole',
        'replay_size': 1000,
        'replay_initial': 64,
        'target_net_sync': 10,
        'epsilon_frames': 5000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 1e-2,
        'gamma': 0.95,
        'batch_size': 32
    })
})



def unpack_batch(batch):
    states,actions,rewards,dones,last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(states)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)
        else:
            last_state = np.array(exp.last_state, copy=False)
            last_states.append(last_state)
    return (np.array(states, copy=False),
            np.array(actions),
            np.array(rewards),
            np.array(dones, dtype=np.bool),
            np.array(last_states, copy=False))

def calc_loss_dqn(batch, net, tgt_net, gamma, device='cpu'):
    states, actions, rewards, dones, last_states = unpack_batch(batch)

    statesV = torch.tensor(states).to(device)
    actionsV = torch.tensor(actions).to(device)
    rewardsV= torch.tensor(rewards).to(device)
    done_tags = torch.BoolTensor(dones).to(device)
    last_statesV = torch.tensor(last_states).to(device)

    state_action_values = net(statesV).gather(dim=1, index=actionsV.unsqueeze(-1)).squeeze(-1)
    max_next_state_values = tgt_net.target_model(last_statesV).max(dim=1)[0]
    max_next_state_values[done_tags] = 0.0
    expected_values = rewardsV + gamma * max_next_state_values.detach()

    return F.mse_loss(state_action_values, expected_values)



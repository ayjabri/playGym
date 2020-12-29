import torch
import numpy as np
import torch.nn.functional as F
import time
import sys
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
        'stop_reward': -30.5,
        'run_name': 'MountainCar',
        'replay_size': 1000,
        'replay_initial': 64,
        'target_net_sync': 1000,
        'epsilon_frames': 50_000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 1e-3,
        'gamma': 0.95,
        'batch_size': 16,
        'step_count': 2,
        'n_envs':1
    })
})

def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        s = np.array(exp.state)
        states.append(s)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(s)
        else:
            last_states.append(np.array(exp.last_state))
    return np.array(states),np.array(actions),np.array(rewards),np.array(dones),np.array(last_states)


def calc_loss_dqn(batch, net, tgt_net, gamma, device='cpu'):
    states,actions,rewards,dones,last_states=unpack_batch(batch)

    states_v = torch.FloatTensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards, dtype=torch.float)
    done_tags = torch.BoolTensor(dones)
    last_states_v = torch.FloatTensor(last_states)

    state_action_values = net(states_v).gather(dim=1, index=actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(last_states_v).max(dim=1)[0].detach()
    next_state_values[done_tags] = 0.0
    expected_values = rewards_v + gamma * next_state_values

    return F.mse_loss(state_action_values, expected_values)


@torch.no_grad()
def play_episode(env, net):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        state_v = torch.tensor([state], dtype=torch.float32)
        action = F.softmax(net(state_v), dim=-1).argmax().item()
        state, reward, done,_=env.step(action)
        rewards += reward
        if done:
            print(rewards)
            time.sleep(1)
            break
    env.close()
    pass


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)




def unpack_a3c_batch(batch, net, last_val_gamma, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v

#%%


import numpy as np
import torch
import ptan
from tensorboardX import SummaryWriter
from lib import common, bolling_wrappers, my_models



params = common.HYPERPARAMS['bolling']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
STEPS =4
ENVS = 1

env = bolling_wrappers.make_bolling('BowlingNoFrameskip-v4')
env = bolling_wrappers.wrap_bolling(env)

net = my_models.DuelDQN(env.observation_space.shape, env.action_space.n).to(device)
tgt_net = ptan.agent.TargetNet(net)

selector = ptan.actions.EpsilonGreedyActionSelector(params.epsilon_start)
agent = ptan.agent.DQNAgent(net, selector, device = device)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, params.gamma, steps_count=STEPS)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.replay_size)
epsilon_tracker = ptan.actions.EpsilonTracker(selector, params.epsilon_start, params.epsilon_final, params.epsilon_frames)
optimizer = torch.optim.SGD(net.parameters(), lr = params.learning_rate)

writer = SummaryWriter(comment=params.run_name)

frame_idx = 0
episode = 0
total_rewards = []

# %%
if __name__ == '__main__':
    
    with ptan.common.utils.RewardTracker(writer) as reward_tracker:
        while True:
            frame_idx += 1
            epsilon_tracker.frame(frame_idx)
            buffer.populate(1)
            
            done_reward = exp_source.pop_total_rewards()
            
            if done_reward:
                mean_rewards = reward_tracker.reward(done_reward[0], frame_idx, selector.epsilon)
                if mean_rewards is not None:
                    if  mean_rewards > params.stop_reward:
                        print('Solved')
                        tgt_net.sync()
                        torch.save(tgt_net.target_model.state_dict(), params.run_name+f'_S{STEPS}_E{ENVS}.dat')
                        break
        
            
            if len(buffer) < params.replay_initial:
                continue
            
            optimizer.zero_grad()
            batch = buffer.sample(params.batch_size * ENVS)
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, params.gamma**STEPS , device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params.target_net_sync == 0:
                tgt_net.sync()
                torch.save(tgt_net.target_model.state_dict(), params.run_name+f'_S{STEPS}_E{ENVS}.dat')
                for name, p in net.named_parameters():
                    writer.add_histogram(name, p)
            

# %%

<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:02:44 2021

@author: Ayman Jabri
"""

import torch
import math
import numpy as np
from lib import model


class MCTS:
    def __init__(self, game, c_pcut = 1.0):
        
        self.game = game
        self.c_pcut = c_pcut
        self.visit_count = {}
        self.value = {}
        self.value_avg = {}
        self.prob = {}
    
    def __len__(self):
        return len(self.value)
    
    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.prob.clear()
        
    def is_leaf(self,state_int):
        return state_int not in self.prob
    
    def find_leaf(self, state_int, player):
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None
        
        while not self.is_leaf(cur_state):
            states.append(cur_state)
            
            counts = self.visit_count[cur_state]
            total_sum = math.sqrt(sum(counts))
            value_avg = self.value_avg[cur_state]
            probs = self.prob[cur_state]
            
            if cur_state == state_int:
                noises = np.random.dirichlet([0.03]*self.game.cols)
                probs = [0.75 * prob + 0.25 * noise for prob,noise in zip(probs, noises)]
            
            score = [value + self.c_pcut * prob * total_sum/(1+count) for \
                     value,prob,count in zip(value_avg,probs,counts)]
            
            invalid_actions = set(range(self.game.cols)) - set(self.game.possible_moves(cur_state))
            for invalid in invalid_actions:
                score[invalid] = - np.inf
            
            action = int(np.argmax(score))
            actions.append(action)
            
            cur_state, won = self.game.move(cur_state, action, cur_player)
            cur_player = 1 - cur_player
            if won:
                value = -1.0
            
            if value is None and len(self.game.possible_moves(cur_state))==0:
                value = 0.0
        
        return value, cur_state, cur_player, states, actions
    
    @torch.no_grad()
    def search_minibatch(self, batch_size, state_int, player, net, device='cpu'):
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()
        
        for _ in range(batch_size):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(state_int, player)
            if value is not None:
                backup_queue.append((value,states,actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_list = self.game.decode_binary(leaf_state)
                    expand_states.append(leaf_state_list)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state,states,actions))
        
        # expand leaf nodes if any
        if expand_queue:
            batch = model.state_lists_to_batch(expand_states, expand_players, device)
            logits_v, values_v = net(batch)
            probs = torch.softmax(logits_v, dim=1).cpu().data.numpy()
            values = values_v.cpu().data.numpy()[:,0]
            
            for (leaf_state,states,actions),value,prob in zip(expand_queue,values,probs):
                self.visit_count[leaf_state] = [0] * self.game.cols
                self.value[leaf_state] = [0.0] * self.game.cols
                self.value_avg[leaf_state] = [0.0] * self.game.cols
                self.prob[leaf_state] = prob
                backup_queue.append((value, states, actions))
        
        # backup values:
        for value,states,actions in backup_queue:
            # reverse the value sign 
            cur_value = -value
            # update all states' values, count and value averages in reversed order
            for state,action in zip(states[::-1],actions[::-1]):
                self.visit_count[state][action] += 1
                self.value[state][action] += cur_value
                self.value_avg[state][action] = self.value[state][action] / self.visit_count[state][action]
                cur_value = -cur_value
    
    def search_batch(self, count, batch_size, state_int, player, net, device='cpu'):
        for _ in range(count):
            self.search_minibatch(batch_size, state_int, player, net, device)
            
    
    def get_policy_value(self, state_int, tau=1.0):
        counts = self.visit_count[state_int]
        values = self.value_avg[state_int]
        if tau ==0.0:
            probs = [0.0] * self.game.cols
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count**(1/tau) for count in counts]
            total = sum(counts)
            probs = [count/total for count in counts]
        return probs, values


def play_game(game, mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first=True, steps_before_tau=None, device='cpu'):
    
    cur_state = game.init_state
    net = [net1,net2]
    cur_player = 0 if white_player_first else 1
    result = None
    result_1 = None
    game_history = []
    steps = 0
    
    while result is None:
        steps += 1
        mcts_stores[cur_player].search_batch(search_count, batch_size, cur_state,cur_player,net[cur_player],device)
        probs,_ = mcts_stores[cur_player].get_policy_value(cur_state)
        action = int(np.argmax(probs))
        cur_state, won = game.move(cur_state, action, cur_player)
        game_history.append((cur_state, cur_player, probs))
        if won:
            print(f'Player {cur_player} won in {steps} steps!!')
            result = 1.0 if cur_player==0 else -1.0
            result_1 = 1.0 if cur_player==0 else -1.0
            break
        if len(game.possible_moves(cur_state))==0:
            print('Draw')
            result_1 = result = 0.0
            break
        cur_player = 1-cur_player
        
    if buffer is not None:
        for state_int, player, probs in reversed(game_history):
            buffer.append((state_int, player, probs, result))
            result *= -1
    
    return result_1, steps

def play_round(n,mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first=True, steps_before_tau=None, device='cpu'):
    player0 = player1 = 0
    for _ in range(n):
        r,_=play_game(mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first, steps_before_tau, device)
        if r > 0:
            player0+=1
        elif r < 0:
            player1+=1
    print(f'player 0 has won: {player0/n *100:.2f}% of the games')



def play_game_net(game, mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first=True, steps_before_tau=None, device='cpu'):
    
    cur_state = game.init_state
    net = [net1,net2]
    cur_player = 0 if white_player_first else 1
    result = None
    result_1 = None
    game_history = []
    steps = 0
    
    while result is None:
        steps += 1
        mcts_stores[cur_player].search_batch(10,10,cur_state,cur_player,net[cur_player],device)
        probs,_ = mcts_stores[cur_player].get_policy_value(cur_state)
        action = int(np.argmax(probs))
        cur_state, won = game.move(cur_state, action, cur_player)
        game_history.append((cur_state, cur_player, probs))
        if won:
            print(f'Player {cur_player} won in {steps} steps!!')
            result = 1.0 if cur_player==0 else -1.0
            result_1 = 1.0 if cur_player==0 else -1.0
            break
        if len(game.possible_moves(cur_state))==0:
            print('Draw')
            result_1 = result = 0.0
            break
        cur_player = 1-cur_player
        
    for state_int, player, probs in reversed(game_history):
        buffer.append((state_int, player, probs, result))
        result *= -1
    
    return result_1, steps    
=======
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:02:44 2021

@author: Ayman Jabri
"""

import torch
import math
import numpy as np
# from lib import mymodel


class MCTS:
    def __init__(self, game, c_pcut = 1.0):

        self.game = game
        self.c_pcut = c_pcut
        self.visit_count = {}
        self.value = {}
        self.value_avg = {}
        self.prob = {}

    def __len__(self):
        return len(self.value)

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.prob.clear()

    def is_leaf(self,state_int):
        return state_int not in self.prob

    def find_leaf(self, state_int, player):
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sum = math.sqrt(sum(counts))
            value_avg = self.value_avg[cur_state]
            probs = self.prob[cur_state]

            if cur_state == state_int:
                noises = np.random.dirichlet([0.03]*self.game.cols)
                probs = [0.75 * prob + 0.25 * noise for prob,noise in zip(probs, noises)]

            score = [value + self.c_pcut * prob * total_sum/(1+count) for \
                     value,prob,count in zip(value_avg,probs,counts)]

            invalid_actions = set(range(self.game.cols)) - set(self.game.possible_moves(cur_state))
            for invalid in invalid_actions:
                score[invalid] = - np.inf

            action = int(np.argmax(score))
            actions.append(action)

            cur_state, won = self.game.move(cur_state, action, cur_player)
            cur_player = 1 - cur_player
            if won:
                value = -1.0

            if value is None and len(self.game.possible_moves(cur_state))==0:
                value = 0.0

        return value, cur_state, cur_player, states, actions

    @torch.no_grad()
    def search_minibatch(self, batch_size, state_int, player, net, device='cpu'):
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()

        for _ in range(batch_size):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(state_int, player)
            if value is not None:
                backup_queue.append((value,states,actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_list = self.game.decode_binary(leaf_state)
                    expand_states.append(leaf_state_list)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state,states,actions))

        # expand leaf nodes if any
        if expand_queue:
            batch = model.state_lists_to_batch(expand_states, expand_players, device)
            logits_v, values_v = net(batch)
            probs = torch.softmax(logits_v, dim=1).cpu().data.numpy()
            values = values_v.cpu().data.numpy()[:,0]

            for (leaf_state,states,actions),value,prob in zip(expand_queue,values,probs):
                self.visit_count[leaf_state] = [0] * self.game.cols
                self.value[leaf_state] = [0.0] * self.game.cols
                self.value_avg[leaf_state] = [0.0] * self.game.cols
                self.prob[leaf_state] = prob
                backup_queue.append((value, states, actions))

        # backup values:
        for value,states,actions in backup_queue:
            # reverse the value sign
            cur_value = -value
            # update all states' values, count and value averages in reversed order
            for state,action in zip(states[::-1],actions[::-1]):
                self.visit_count[state][action] += 1
                self.value[state][action] += cur_value
                self.value_avg[state][action] = self.value[state][action] / self.visit_count[state][action]
                cur_value = -cur_value

    def search_batch(self, count, batch_size, state_int, player, net, device='cpu'):
        for _ in range(count):
            self.search_minibatch(batch_size, state_int, player, net, device)


    def get_policy_value(self, state_int, tau=1.0):
        counts = self.visit_count[state_int]
        values = self.value_avg[state_int]
        if tau ==0.0:
            probs = [0.0] * self.game.cols
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count**(1/tau) for count in counts]
            total = sum(counts)
            probs = [count/total for count in counts]
        return probs, values


def play_game(game, mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first=True, steps_before_tau=None, device='cpu'):

    cur_state = game.init_state
    net = [net1,net2]
    cur_player = 0 if white_player_first else 1
    result = None
    result_1 = None
    game_history = []
    steps = 0

    while result is None:
        steps += 1
        mcts_stores[cur_player].search_batch(search_count, batch_size, cur_state,cur_player,net[cur_player],device)
        probs,_ = mcts_stores[cur_player].get_policy_value(cur_state)
        action = int(np.argmax(probs))
        cur_state, won = game.move(cur_state, action, cur_player)
        game_history.append((cur_state, cur_player, probs))
        if won:
            print(f'Player {cur_player} won in {steps} steps!!')
            result = 1.0 if cur_player==0 else -1.0
            result_1 = 1.0 if cur_player==0 else -1.0
            break
        if len(game.possible_moves(cur_state))==0:
            print('Draw')
            result_1 = result = 0.0
            break
        cur_player = 1-cur_player

    if buffer is not None:
        for state_int, player, probs in reversed(game_history):
            buffer.append((state_int, player, probs, result))
            result *= -1

    return result_1, steps

def play_round(n,mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first=True, steps_before_tau=None, device='cpu'):
    player0 = player1 = 0
    for _ in range(n):
        r,_=play_game(mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first, steps_before_tau, device)
        if r > 0:
            player0+=1
        elif r < 0:
            player1+=1
    print(f'player 0 has won: {player0/n *100:.2f}% of the games')



def play_game_net(game, mcts_stores, buffer, search_count, batch_size, net1, net2,
              white_player_first=True, steps_before_tau=None, device='cpu'):

    cur_state = game.init_state
    net = [net1,net2]
    cur_player = 0 if white_player_first else 1
    result = None
    result_1 = None
    game_history = []
    steps = 0

    while result is None:
        steps += 1
        mcts_stores[cur_player].search_batch(10,10,cur_state,cur_player,net[cur_player],device)
        probs,_ = mcts_stores[cur_player].get_policy_value(cur_state)
        action = int(np.argmax(probs))
        cur_state, won = game.move(cur_state, action, cur_player)
        game_history.append((cur_state, cur_player, probs))
        if won:
            print(f'Player {cur_player} won in {steps} steps!!')
            result = 1.0 if cur_player==0 else -1.0
            result_1 = 1.0 if cur_player==0 else -1.0
            break
        if len(game.possible_moves(cur_state))==0:
            print('Draw')
            result_1 = result = 0.0
            break
        cur_player = 1-cur_player

    for state_int, player, probs in reversed(game_history):
        buffer.append((state_int, player, probs, result))
        result *= -1

    return result_1, steps
>>>>>>> 407adb5018e7855d1d4d69c45901cda70c32f9aa

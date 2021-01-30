# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:39:53 2021

@author: ayman
"""
import argparse
import random
import numpy as np
from lib import mcts, model
from typing import Optional, List, Tuple
from lib.game import game


def get_action(game, net, mcts_, cur_state,cur_player, search_depth, batch_size):
    if net is None:
        action = int(input('Enter column number to place a stone in (0->6): '))
        return action
    mcts_.search_batch(search_depth, batch_size, cur_state, cur_player, net)
    prob, _ = mcts_.get_policy_value(cur_state)
    return int(np.argmax(prob))



def play(net1,net2=None,mcts_store=None, batch_size=16,
         search_depth=10,steps_before_tau=10,device='cpu'):
    if net2 is None:
        print('*'*10,' Human vs Computer', '*'*10)
        print('\t    You are the Black player  \n')
    nets = [net1, net2]
    connect = game.Connect()
    if mcts_store is None:
        mcts_store = [mcts.MCTS(connect), mcts.MCTS(connect)]
    cur_player = random.choice(range(2))
    cur_state = connect.init_state
    player = 'White' if cur_player ==0 else 'Black'
    print(f'########## {player} plays first ##########\n\n')
    while True:
        connect.print_board(cur_state)
        action = get_action(connect,
                            nets[cur_player],
                            mcts_store[cur_player],
                            cur_state,
                            cur_player,
                            search_depth,
                            batch_size
                            )
        cur_state, won = connect.move(cur_state, action, cur_player)
        if won:
            connect.print_board(cur_state)
            print(f'Player {cur_player} wins')
            break
        cur_player = 1-cur_player


if __name__=='__main__':
    parser = argparse.PARSER()
    parser.add_argument('-net1', default=None, help='path to net1 dict_state')
    args = parser.parse_args()
    net = model.Net()
    play(net)